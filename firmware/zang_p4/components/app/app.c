/* app.c - ZLink dispatcher, state machine glue, STATE publisher. */
#include "app.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "app_internal.h"
#include "app_sm.h"
#include "board.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "net.h"
#include "storage.h"
#include "zang_tasks.h"
#include "zang_units.h"
#include "zlink.h"
#include "zlink_server.h"

static_assert(ROBOT_N_JOINTS == ZLINK_N_JOINTS, "robot.json joint count must match ZLink v1 (15)");

static const char *TAG = "app";

static struct {
    app_ctx_t c;
    zlink_server_handle_t srv;
    SemaphoreHandle_t lock;
    app_state_t state;
    uint8_t state_hz;
    uint32_t audio_stream_id;
    bool audio_stream_open;
    bool opus_warned;
    esp_timer_handle_t audio_start_timer;
    char pending_audio[96];
} s;

/* ------------------------------------------------------------------ helpers */

uint8_t app_err_to_status(esp_err_t e)
{
    switch (e) {
    case ESP_OK: return 0;
    case ESP_ERR_INVALID_STATE: return 1;
    case ESP_ERR_INVALID_ARG: return 2;
    case ESP_ERR_NOT_FOUND: return 3;
    case ESP_ERR_NOT_SUPPORTED: return 4;
    default: return 5;
    }
}

void app_send_event(uint8_t kind, const uint8_t *data, uint16_t n)
{
    uint8_t p[64];
    uint16_t len;
    zlink_event_t ev = {.kind = kind, .data = data, .n_data = n};
    if (zlink_pack_event(&ev, p, sizeof(p), &len) != ZLINK_OK) return;
    /* events go to the UDP peer; a TCP-only client (bulk upload) gets them on its connection */
    if (zlink_server_send(s.srv, ZLINK_VIA_UDP, ZLINK_EVENT, p, len) == ESP_ERR_INVALID_STATE) {
        zlink_server_send(s.srv, ZLINK_VIA_TCP, ZLINK_EVENT, p, len);
    }
}

void app_send(zlink_via_t via, uint8_t type, const uint8_t *payload, uint16_t len)
{
    zlink_server_send(s.srv, via, type, payload, len);
}

bool app_has_peer(void) { return zlink_server_has_peer(s.srv); }
const app_ctx_t *app_ctx(void) { return &s.c; }

static void show_state_locked(void)
{
    if (!s.c.display) return;
    static const display_face_t faces[APP_STATE_COUNT] = {
        [APP_BOOT] = DISPLAY_FACE_THINKING, [APP_IDLE] = DISPLAY_FACE_SLEEPY, [APP_HOLD] = DISPLAY_FACE_NEUTRAL,
        [APP_STREAMING] = DISPLAY_FACE_HAPPY, [APP_PLAYING_FILE] = DISPLAY_FACE_HAPPY, [APP_ESTOP] = DISPLAY_FACE_ERROR,
    };
    if (!app_host_owns_face() || s.state == APP_ESTOP) display_set_face(s.c.display, faces[s.state]);
    char line[40];
    const char *ip = net_ip_str();
    snprintf(line, sizeof(line), "%s %s", app_state_name(s.state), ip[0] ? ip : "no wifi");
    display_set_status(s.c.display, line);
}

/* Apply a state-machine event; logs + notifies on change. Returns whether it was accepted. */
static bool sm(app_event_t ev)
{
    xSemaphoreTake(s.lock, portMAX_DELAY);
    app_state_t before = s.state;
    bool ok = app_sm_step(&s.state, ev);
    app_state_t after = s.state;
    if (after != before) show_state_locked();
    xSemaphoreGive(s.lock);
    if (after != before) {
        ESP_LOGI(TAG, "state %s -> %s", app_state_name(before), app_state_name(after));
        uint8_t d = (uint8_t)after;
        app_send_event(APP_EVENT_STATE_CHANGED, &d, 1);
    }
    return ok;
}

static app_state_t state_now(void)
{
    xSemaphoreTake(s.lock, portMAX_DELAY);
    app_state_t st = s.state;
    xSemaphoreGive(s.lock);
    return st;
}

static void q_from_cdeg(const int16_t *cdeg, float *q)
{
    for (int i = 0; i < ZLINK_N_JOINTS; i++) q[i] = cdeg[i] == ZLINK_CDEG_NA ? NAN : zang_cdeg_to_rad(cdeg[i]);
}

/* ------------------------------------------------------------------ motion events */

static void on_motion_event(motion_event_t ev, void *ctx)
{
    switch (ev) {
    case MOTION_EV_STREAM_DONE:
        sm(APP_EV_STREAM_DONE);
        app_send_event(APP_EVENT_STREAM_DONE, NULL, 0);
        break;
    case MOTION_EV_FILE_DONE:
        sm(APP_EV_FILE_DONE);
        app_send_event(APP_EVENT_FILE_DONE, NULL, 0);
        break;
    case MOTION_EV_WATCHDOG:
        if (s.c.audio) audio_stop(s.c.audio);
        sm(APP_EV_WATCHDOG);
        app_send_event(APP_EVENT_WATCHDOG, NULL, 0);
        break;
    case MOTION_EV_FAULT: {
        motion_state_t ms;
        motion_get_state(s.c.motion, &ms);
        app_send_event(APP_EVENT_SERVO_FAULT, &ms.fault_servo_id, 1);
        break;
    }
    case MOTION_EV_ESTOP:
        if (s.c.audio) audio_stop(s.c.audio);
        sm(APP_EV_ESTOP);
        app_send_event(APP_EVENT_ESTOP, NULL, 0);
        break;
    case MOTION_EV_TARGET_DONE:
        app_send_event(APP_EVENT_TARGET_DONE, NULL, 0);
        break;
    }
}

static bool audio_clock(void *ctx, uint32_t *ms) { return s.c.audio && audio_clock_ms(s.c.audio, ms); }

static void start_pending_audio(void *arg)
{
    if (s.c.audio && s.pending_audio[0]) audio_play_wav(s.c.audio, s.pending_audio);
}

/* ------------------------------------------------------------------ commands */

static void reply(zlink_via_t via, uint8_t type, const uint8_t *p, uint16_t len);

static void send_calib_report(zlink_via_t via)
{
    motion_calib_t cal;
    if (!s.c.motion || motion_calib_get(s.c.motion, &cal) != ESP_OK) return;
    zlink_calib_report_t r;
    for (int i = 0; i < ZLINK_N_JOINTS; i++) {
        r.joints[i].zero_ticks = cal.j[i].zero_ticks;
        r.joints[i].sign = cal.j[i].sign;
        r.joints[i].ticks_per_rev = cal.j[i].ticks_per_rev;
    }
    uint8_t p[5 * ZLINK_N_JOINTS];
    uint16_t len;
    if (zlink_pack_calib_report(&r, p, sizeof(p), &len) == ZLINK_OK) reply(via, ZLINK_CALIB_REPORT, p, len);
}

static esp_err_t do_cmd(const zlink_cmd_t *cmd, zlink_via_t via)
{
    esp_err_t media = app_media_cmd(cmd); /* MIC_STREAM, CAMERA_STREAM, SET_VOLUME, SET_BRIGHTNESS, SNAPSHOT */
    if (media != ESP_ERR_NOT_FOUND) return media;
    if (!s.c.motion) return ESP_ERR_NOT_SUPPORTED;
    switch (cmd->cmd) {
    case ZLINK_CMD_HOLD:
        if (s.c.audio) audio_stop(s.c.audio);
        if (!sm(APP_EV_HOLD)) return ESP_ERR_INVALID_STATE;
        return motion_hold(s.c.motion);
    case ZLINK_CMD_ESTOP:
        if (s.c.audio) audio_stop(s.c.audio);
        sm(APP_EV_ESTOP);
        return motion_estop(s.c.motion);
    case ZLINK_CMD_TORQUE_ON: {
        if (!s.c.board_ok) return ESP_ERR_INVALID_STATE;
        esp_err_t err = motion_torque_on(s.c.motion);
        if (err == ESP_OK) sm(APP_EV_TORQUE_ON);
        return err;
    }
    case ZLINK_CMD_GO_HOME: {
        uint16_t ms;
        if (zlink_cmd_go_home_ms(cmd, &ms) != ZLINK_OK) return ESP_ERR_INVALID_ARG;
        if (!app_sm_can_move(state_now())) return ESP_ERR_INVALID_STATE;
        sm(APP_EV_HOLD);
        return motion_go_home(s.c.motion, ms);
    }
    case ZLINK_CMD_PLAY_FILE: {
        char rel[64], path[96], audio_name[ZACT_AUDIO_LEN + 1];
        if (zlink_cmd_play_file_path(cmd, rel, sizeof(rel)) != ZLINK_OK) return ESP_ERR_INVALID_ARG;
        if (!storage_is_mounted()) return ESP_ERR_NOT_FOUND;
        if (!app_sm_can_move(state_now())) return ESP_ERR_INVALID_STATE;
        if (rel[0] == '/') snprintf(path, sizeof(path), "%s", rel);
        else snprintf(path, sizeof(path), STORAGE_ROOT "/%s", rel);
        uint32_t audio_at;
        esp_err_t err = motion_play_file(s.c.motion, path, audio_name, sizeof(audio_name), &audio_at);
        if (err != ESP_OK) return err;
        sm(APP_EV_PLAY_FILE);
        if (audio_name[0] && s.c.audio) { /* companion audio lives next to the .zact */
            char *slash = strrchr(path, '/');
            int dir_len = slash ? (int)(slash - path) : 0;
            snprintf(s.pending_audio, sizeof(s.pending_audio), "%.*s/%s", dir_len, path, audio_name);
            int64_t delay_us = ((int64_t)audio_at - (int64_t)motion_now_ms()) * 1000;
            esp_timer_stop(s.audio_start_timer);
            esp_timer_start_once(s.audio_start_timer, delay_us > 0 ? (uint64_t)delay_us : 1);
        }
        return ESP_OK;
    }
    case ZLINK_CMD_SET_RATE: {
        uint8_t hz;
        if (zlink_cmd_set_rate_hz(cmd, &hz) != ZLINK_OK || hz == 0 || hz > 100) return ESP_ERR_INVALID_ARG;
        s.state_hz = hz;
        return ESP_OK;
    }
    case ZLINK_CMD_CALIB_SET: {
        zlink_calib_set_t c;
        if (zlink_cmd_calib_set(cmd, &c) != ZLINK_OK) return ESP_ERR_INVALID_ARG;
        return motion_calib_set(s.c.motion, c.index, c.zero_ticks, c.sign, c.has_ticks_per_rev ? c.ticks_per_rev : 0);
    }
    case ZLINK_CMD_CALIB_GET:
        send_calib_report(via); /* CALIB_REPORT first, then the ACK */
        return ESP_OK;
    case ZLINK_CMD_CALIB_SAVE:
        return motion_calib_save(s.c.motion);
    default:
        return ESP_ERR_NOT_SUPPORTED;
    }
}

/* ------------------------------------------------------------------ ZLink dispatch */

static void reply(zlink_via_t via, uint8_t type, const uint8_t *p, uint16_t len)
{
    zlink_server_send(s.srv, via, type, p, len);
}

static void on_frame(const zlink_frame_t *f, zlink_via_t via, void *ctx)
{
    uint8_t out[64];
    uint16_t len;
    if (s.c.motion) motion_keepalive(s.c.motion);
    switch (f->type) {
    case ZLINK_HELLO: {
        zlink_hello_ack_t ack = {.proto_ver = ZLINK_VERSION, .n_joints = ZLINK_N_JOINTS,
                                 .fw_ver = (uint16_t)(CONFIG_ZANG_FW_VERSION & 0x7FFF) | ZLINK_FW_V11_BIT};
        strlcpy(ack.name, ROBOT_NAME, sizeof(ack.name));
        if (zlink_pack_hello_ack(&ack, out, sizeof(out), &len) == ZLINK_OK) reply(via, ZLINK_HELLO_ACK, out, len);
        break;
    }
    case ZLINK_PING: {
        zlink_ping_t p;
        if (zlink_unpack_ping(f->payload, f->len, &p) != ZLINK_OK) break;
        zlink_pong_t pong = {.host_us = p.host_us, .robot_us = (uint32_t)esp_timer_get_time()};
        if (zlink_pack_pong(&pong, out, sizeof(out), &len) == ZLINK_OK) reply(via, ZLINK_PONG, out, len);
        break;
    }
    case ZLINK_JOINT_TARGETS: {
        zlink_joint_targets_t t;
        if (zlink_unpack_joint_targets(f->payload, f->len, &t) != ZLINK_OK || !s.c.motion) break;
        if (!app_sm_can_move(state_now())) break;
        float q[ZLINK_N_JOINTS];
        q_from_cdeg(t.q_cdeg, q);
        if (motion_set_targets(s.c.motion, t.t_robot_ms, t.mask, t.move_ms, q) == ESP_OK) sm(APP_EV_HOLD);
        break;
    }
    case ZLINK_TRAJ_CHUNK: {
        static zlink_traj_chunk_t ch; /* 1.3 kB: static, only the transport tasks get here (one at a time) */
        static uint16_t dt[ZLINK_TRAJ_MAX_FRAMES];
        static float q[ZLINK_TRAJ_MAX_FRAMES][ZLINK_N_JOINTS];
        if (zlink_unpack_traj_chunk(f->payload, f->len, &ch) != ZLINK_OK || !s.c.motion) break;
        for (int k = 0; k < ch.n_frames; k++) {
            dt[k] = ch.frames[k].dt_ms;
            q_from_cdeg(ch.frames[k].q_cdeg, q[k]);
        }
        motion_stream_push(s.c.motion, ch.stream_id, ch.t0_ms, (motion_interp_t)ch.interp, dt,
                           (const float (*)[MOTION_N])q, ch.n_frames);
        break;
    }
    case ZLINK_STREAM_START: {
        zlink_stream_start_t st;
        if (zlink_unpack_stream_start(f->payload, f->len, &st) != ZLINK_OK || !s.c.motion) break;
        if (!app_sm_can_move(state_now())) break;
        if (motion_stream_start(s.c.motion, st.stream_id, st.clock == ZLINK_CLOCK_AUDIO) == ESP_OK) {
            s.audio_stream_open = false;
            sm(APP_EV_STREAM_START);
        }
        break;
    }
    case ZLINK_STREAM_END: {
        zlink_stream_end_t e;
        if (zlink_unpack_stream_end(f->payload, f->len, &e) != ZLINK_OK) break;
        if (s.c.motion) motion_stream_end(s.c.motion, e.stream_id);
        if (s.c.audio && s.audio_stream_open) audio_stream_end(s.c.audio, e.stream_id);
        break;
    }
    case ZLINK_AUDIO_CHUNK: {
        zlink_audio_chunk_t a;
        if (zlink_unpack_audio_chunk(f->payload, f->len, &a) != ZLINK_OK || !s.c.audio) break;
        if (a.codec == ZLINK_CODEC_OPUS) {
            if (!s.opus_warned) ESP_LOGW(TAG, "Opus audio not supported yet: send PCM16");
            s.opus_warned = true;
            break;
        }
        uint32_t rate = a.codec == ZLINK_CODEC_PCM16_24K ? 24000 : 16000;
        if (!s.audio_stream_open || s.audio_stream_id != a.stream_id) {
            if (audio_stream_begin(s.c.audio, a.stream_id, rate) != ESP_OK) break;
            s.audio_stream_open = true;
            s.audio_stream_id = a.stream_id;
        }
        audio_stream_write(s.c.audio, a.stream_id, a.sample_offset, (const int16_t *)a.data, a.n_data / 2);
        break;
    }
    case ZLINK_CMD: {
        zlink_cmd_t cmd;
        if (zlink_unpack_cmd(f->payload, f->len, &cmd) != ZLINK_OK) break;
        esp_err_t err = do_cmd(&cmd, via);
        if (err != ESP_OK) ESP_LOGW(TAG, "CMD %u failed: %s", cmd.cmd, esp_err_to_name(err));
        zlink_ack_t ack = {.seq_acked = f->seq, .status = app_err_to_status(err)};
        if (zlink_pack_ack(&ack, out, sizeof(out), &len) == ZLINK_OK) reply(via, ZLINK_ACK, out, len);
        break;
    }
    default:
        app_media_on_frame(f, via); /* FACE_*, DISPLAY_* */
        break;
    }
}

/* ------------------------------------------------------------------ STATE publisher */

static void state_task(void *arg)
{
    TickType_t wake = xTaskGetTickCount();
    for (;;) {
        uint8_t hz = s.state_hz ? s.state_hz : 50;
        vTaskDelayUntil(&wake, pdMS_TO_TICKS(1000 / hz));
        if (!zlink_server_has_peer(s.srv)) continue;
        zlink_state_t st = {.t_robot_ms = (uint32_t)(esp_timer_get_time() / 1000)};
        if (s.c.motion) {
            motion_state_t ms;
            motion_get_state(s.c.motion, &ms);
            for (int i = 0; i < ZLINK_N_JOINTS; i++) {
                st.q_meas_cdeg[i] = isnan(ms.q_meas[i]) ? ZLINK_CDEG_NA : zang_rad_to_cdeg(ms.q_meas[i]);
                st.q_cmd_cdeg[i] = zang_rad_to_cdeg(ms.q_cmd[i]);
            }
            if (ms.torque_on) st.flags |= ZLINK_STATE_TORQUE_ON;
            if (ms.streaming) st.flags |= ZLINK_STATE_STREAMING;
            if (ms.calibrated) st.flags |= ZLINK_STATE_CALIBRATED;
            if (ms.fault_servo_id) st.flags |= ZLINK_STATE_FAULT;
            st.fault_servo_id = ms.fault_servo_id;
            st.stream_buffered_ms = (uint16_t)ms.stream_buffered_ms;
            st.stream_pos_ms = ms.stream_pos_ms;
        } else {
            for (int i = 0; i < ZLINK_N_JOINTS; i++) st.q_meas_cdeg[i] = st.q_cmd_cdeg[i] = ZLINK_CDEG_NA;
        }
        if (state_now() == APP_ESTOP) st.flags |= ZLINK_STATE_ESTOP;
        imu_state_t im;
        if (s.c.imu && imu_get(s.c.imu, &im) == ESP_OK && im.valid) {
            for (int i = 0; i < 3; i++) {
                st.rpy_cdeg[i] = zang_rad_to_cdeg(im.rpy_rad[i]);
                st.gyro_ddps[i] = (int16_t)lroundf(im.gyro_dps[i] * 10.0f);
                st.acc_mg[i] = (int16_t)lroundf(im.acc_g[i] * 1000.0f);
            }
        } else { /* no IMU on board v0.0.1 */
            for (int i = 0; i < 3; i++) st.rpy_cdeg[i] = st.gyro_ddps[i] = st.acc_mg[i] = ZLINK_CDEG_NA;
        }
        st.vbat_mv = board_vbat_mv();
        st.display_mode = app_media_display_mode();
        st.emotion = app_media_emotion();
        uint8_t p[ZLINK_STATE_PAYLOAD_LEN];
        uint16_t len;
        if (zlink_pack_state(&st, p, sizeof(p), &len) == ZLINK_OK) {
            zlink_server_send(s.srv, ZLINK_VIA_UDP, ZLINK_STATE, p, len);
        }
    }
}

/* Stream playhead for FACE_TRACK (called from the display task). */
static bool stream_source(void *ctx, uint32_t *stream_id, uint32_t *pos_ms)
{
    motion_state_t ms;
    if (!s.c.motion || motion_get_state(s.c.motion, &ms) != ESP_OK || !ms.streaming) return false;
    *stream_id = ms.stream_id;
    *pos_ms = ms.stream_pos_ms;
    return true;
}

esp_err_t app_start(const app_ctx_t *ctx)
{
    ESP_RETURN_ON_FALSE(ctx, ESP_ERR_INVALID_ARG, TAG, "null");
    s.c = *ctx;
    s.lock = xSemaphoreCreateMutex();
    s.state = APP_BOOT;
    s.state_hz = CONFIG_ZANG_STATE_HZ;
    ESP_RETURN_ON_FALSE(s.lock, ESP_ERR_NO_MEM, TAG, "lock");
    const esp_timer_create_args_t ta = {.callback = start_pending_audio, .name = "audio_start"};
    ESP_RETURN_ON_ERROR(esp_timer_create(&ta, &s.audio_start_timer), TAG, "timer");

    zlink_server_config_t zc = {.udp_port = ZLINK_UDP_PORT, .tcp_port = ZLINK_TCP_PORT, .on_frame = on_frame};
    esp_err_t err = zlink_server_start(&zc, &s.srv);
    if (err != ESP_OK) ESP_LOGE(TAG, "ZLink server failed: %s", esp_err_to_name(err));
    if (s.srv && zang_task_create(ZANG_TASK_STATE_TX, state_task, NULL, NULL) != pdPASS) return ESP_ERR_NO_MEM;
    if (s.c.display) display_set_stream_source(s.c.display, stream_source, NULL);
    app_media_start(&s.c);
    sm(ctx->board_ok ? APP_EV_BOOT_DONE : APP_EV_ESTOP);
    show_state_locked();
    return ESP_OK;
}

/* Called by main to hook motion events and the audio clock before motion_start(). */
void app_motion_config(motion_config_t *cfg)
{
    cfg->on_event = on_motion_event;
    cfg->audio_clock = audio_clock;
    cfg->ctx = NULL;
}

void app_set_audio(audio_handle_t audio) { s.c.audio = audio; }
