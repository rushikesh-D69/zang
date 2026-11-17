/* motion.c - real-time motion runtime. See motion.h. */
#include "motion.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "board.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "zang_tasks.h"

static const char *TAG = "motion";

#define PERIOD_MS        CONFIG_ZANG_MOTION_PERIOD_MS
#define READ_FAIL_FAULT  5   /* consecutive failed reads before a servo is reported as faulty */
#define FILE_BATCH       16  /* .zact frames read per cycle at most */

#ifndef CONFIG_ZANG_MOTION_ALLOW_UNCALIBRATED
#define CONFIG_ZANG_MOTION_ALLOW_UNCALIBRATED 0
#endif

struct motion {
    motion_config_t cfg;
    SemaphoreHandle_t lock;

    /* --- protected by lock --- */
    motion_mode_t mode;
    bool torque_on;
    motion_traj_t traj;     /* TARGET / FILE: robot clock; STREAM: stream clock */
    float q_hold[MOTION_N];
    /* stream */
    uint32_t stream_id;
    bool stream_ending;     /* STREAM_END received */
    bool stream_audio_clock;
    int64_t stream_origin_ms; /* robot time at stream time 0 (robot-clock streams) */
    int64_t last_rx_ms;
    /* file */
    FILE *file;
    zact_header_t file_hdr;
    uint32_t file_frames_left;
    int64_t file_origin_ms;
    /* outputs */
    motion_calib_t calib;
    motion_limiter_t lim;
    float q_cmd[MOTION_N];
    float q_meas[MOTION_N];
    uint8_t read_fail[MOTION_N];
    uint8_t fault_servo_id;
    uint32_t limit_hits;
};

static inline int64_t now_ms64(void) { return esp_timer_get_time() / 1000; }
uint32_t motion_now_ms(void) { return (uint32_t)now_ms64(); }

static void emit(struct motion *m, motion_event_t ev)
{
    if (m->cfg.on_event) m->cfg.on_event(ev, m->cfg.ctx);
}

/* Caller holds lock. Close any open file. */
static void file_close_locked(struct motion *m)
{
    if (m->file) {
        fclose(m->file);
        m->file = NULL;
    }
}

/* Caller holds lock. Enter HOLD at the current commanded pose. */
static void hold_locked(struct motion *m)
{
    file_close_locked(m);
    memcpy(m->q_hold, m->q_cmd, sizeof(m->q_hold));
    m->mode = m->torque_on ? MOTION_MODE_HOLD : MOTION_MODE_OFF;
    m->stream_ending = false;
}

/* Expand a robot time (u32 ms since boot, as sent in ZLink) to the 64-bit robot clock near now. */
static int64_t robot_ms_from_u32(uint32_t t)
{
    int64_t now = now_ms64();
    int64_t base = now & ~(int64_t)0xFFFFFFFF;
    int64_t cand = base | t;
    if (cand - now > 0x80000000LL) cand -= 0x100000000LL;
    else if (now - cand > 0x80000000LL) cand += 0x100000000LL;
    return cand;
}

/* Caller holds lock. Current stream time in ms. */
static int64_t stream_now_locked(struct motion *m)
{
    uint32_t a;
    if (m->stream_audio_clock && m->cfg.audio_clock && m->cfg.audio_clock(m->cfg.ctx, &a)) return (int64_t)a;
    return now_ms64() - m->stream_origin_ms;
}

/* Caller holds lock. Move .zact frames into the trajectory while there is room. Returns false at EOF/error. */
static bool file_refill_locked(struct motion *m)
{
    uint8_t buf[ZACT_FRAME_SIZE(MOTION_N)];
    for (int k = 0; k < FILE_BATCH && m->file_frames_left > 0 && motion_traj_room(&m->traj) > 0; k++) {
        if (fread(buf, 1, sizeof(buf), m->file) != sizeof(buf)) {
            ESP_LOGW(TAG, "file truncated, %lu frames missing", (unsigned long)m->file_frames_left);
            m->file_frames_left = 0;
            break;
        }
        uint32_t t;
        int16_t qc[MOTION_N];
        float q[MOTION_N];
        zact_parse_frame(buf, sizeof(buf), &m->file_hdr, &t, qc);
        for (int i = 0; i < MOTION_N; i++) q[i] = (qc[i] == 0x7FFF) ? m->q_cmd[i] : qc[i] * (3.14159265f / 18000.0f);
        if (motion_traj_push(&m->traj, m->file_origin_ms + t, q) != MOTION_TRAJ_OK) {
            ESP_LOGW(TAG, "file frame at %lu ms out of order: skipped", (unsigned long)t);
        }
        m->file_frames_left--;
    }
    return m->file_frames_left > 0;
}

/* Caller holds lock. Compute this cycle's target and handle mode transitions. Returns events to emit. */
static uint32_t target_locked(struct motion *m, float target[MOTION_N])
{
    uint32_t events = 0;
    const int64_t now = now_ms64();
    switch (m->mode) {
    case MOTION_MODE_OFF:
    case MOTION_MODE_HOLD:
        memcpy(target, m->q_hold, sizeof(m->q_hold));
        break;
    case MOTION_MODE_TARGET:
        if (!motion_traj_sample(&m->traj, now, target)) {
            memcpy(m->q_hold, target, sizeof(m->q_hold));
            m->mode = MOTION_MODE_HOLD;
            events |= 1u << MOTION_EV_TARGET_DONE;
        }
        break;
    case MOTION_MODE_STREAM: {
        bool active = motion_traj_sample(&m->traj, stream_now_locked(m), target);
        if (now - m->last_rx_ms > CONFIG_ZANG_MOTION_WATCHDOG_MS) {
            ESP_LOGW(TAG, "stream %lu: no host frame for %d ms -> HOLD", (unsigned long)m->stream_id,
                     CONFIG_ZANG_MOTION_WATCHDOG_MS);
            hold_locked(m);
            memcpy(m->q_hold, target, sizeof(m->q_hold));
            events |= 1u << MOTION_EV_WATCHDOG;
        } else if (!active && m->stream_ending) {
            memcpy(m->q_hold, target, sizeof(m->q_hold));
            m->mode = MOTION_MODE_HOLD;
            m->stream_ending = false;
            events |= 1u << MOTION_EV_STREAM_DONE;
        }
        break;
    }
    case MOTION_MODE_FILE: {
        bool more = file_refill_locked(m);
        if (!motion_traj_sample(&m->traj, now, target) && !more) {
            memcpy(m->q_hold, target, sizeof(m->q_hold));
            file_close_locked(m);
            m->mode = MOTION_MODE_HOLD;
            events |= 1u << MOTION_EV_FILE_DONE;
        }
        break;
    }
    }
    return events;
}

static void motion_task(void *arg)
{
    struct motion *m = arg;
    uint8_t ids[MOTION_N];
    for (int i = 0; i < MOTION_N; i++) ids[i] = robot_joints[i].servo_id;
    int read_idx = 0;
    TickType_t wake = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&wake, pdMS_TO_TICKS(PERIOD_MS));
        float target[MOTION_N], q[MOTION_N];
        uint16_t ticks[MOTION_N];

        xSemaphoreTake(m->lock, portMAX_DELAY);
        uint32_t events = target_locked(m, target);
        bool torque = m->torque_on;
        uint32_t clamped = motion_limiter_step(&m->lim, target, PERIOD_MS / 1000.0f, q);
        if (clamped) m->limit_hits++;
        memcpy(m->q_cmd, q, sizeof(q));
        for (int i = 0; i < MOTION_N; i++) ticks[i] = motion_calib_to_ticks(&m->calib.j[i], q[i], NULL);
        motion_calib_t calib = m->calib; /* copy for use outside the lock */
        xSemaphoreGive(m->lock);

        for (int ev = 0; ev < 32; ev++) {
            if (events & (1u << ev)) emit(m, (motion_event_t)ev);
        }
        if (!m->cfg.bus) continue;
        if (torque && board_estop_pressed()) {
            motion_estop(m);
            emit(m, MOTION_EV_ESTOP);
            continue;
        }
        if (torque) servo_bus_set_goals(m->cfg.bus, ids, ticks, PERIOD_MS, MOTION_N);

        for (int k = 0; k < CONFIG_ZANG_MOTION_READS_PER_CYCLE; k++) {
            int i = read_idx;
            read_idx = (read_idx + 1) % MOTION_N;
            uint16_t pos;
            esp_err_t err = servo_bus_read_position(m->cfg.bus, ids[i], &pos);
            bool fault_now = false;
            xSemaphoreTake(m->lock, portMAX_DELAY);
            if (err == ESP_OK) {
                m->q_meas[i] = motion_calib_to_rad(&calib.j[i], pos);
                m->read_fail[i] = 0;
            } else if (m->read_fail[i] < 255 && ++m->read_fail[i] == READ_FAIL_FAULT) {
                m->q_meas[i] = NAN;
                if (!m->fault_servo_id) {
                    m->fault_servo_id = ids[i];
                    fault_now = true;
                }
            }
            xSemaphoreGive(m->lock);
            if (fault_now) {
                ESP_LOGE(TAG, "servo %u (%s) not answering", ids[i], robot_joints[i].name);
                emit(m, MOTION_EV_FAULT);
            }
        }
    }
}

esp_err_t motion_start(const motion_config_t *cfg, motion_handle_t *out)
{
    ESP_RETURN_ON_FALSE(cfg && out, ESP_ERR_INVALID_ARG, TAG, "null arg");
    struct motion *m = calloc(1, sizeof(*m));
    ESP_RETURN_ON_FALSE(m, ESP_ERR_NO_MEM, TAG, "no mem");
    m->cfg = *cfg;
    m->lock = xSemaphoreCreateMutex();
    if (!m->lock) {
        free(m);
        return ESP_ERR_NO_MEM;
    }
    esp_err_t err = motion_calib_load(&m->calib);
    if (err != ESP_OK) ESP_LOGW(TAG, "no stored calibration (%s): using defaults, robot UNCALIBRATED",
                                esp_err_to_name(err));
    float home[MOTION_N];
    for (int i = 0; i < MOTION_N; i++) {
        home[i] = robot_joints[i].home_rad;
        m->q_meas[i] = NAN;
    }
    memcpy(m->q_cmd, home, sizeof(home));
    memcpy(m->q_hold, home, sizeof(home));
    motion_limiter_reset(&m->lim, home);
    m->mode = MOTION_MODE_OFF;
    if (zang_task_create(ZANG_TASK_MOTION, motion_task, m, NULL) != pdPASS) {
        vSemaphoreDelete(m->lock);
        free(m);
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "started: %d joints, %d ms period, bus %s, calibrated=%d", MOTION_N, PERIOD_MS,
             cfg->bus ? "yes" : "NONE (no hardware)", m->calib.from_nvs);
    *out = m;
    return ESP_OK;
}

esp_err_t motion_torque_on(motion_handle_t m)
{
    ESP_RETURN_ON_FALSE(m, ESP_ERR_INVALID_ARG, TAG, "null");
    if (!m->calib.from_nvs && !CONFIG_ZANG_MOTION_ALLOW_UNCALIBRATED) {
        ESP_LOGE(TAG, "refusing torque: robot is not calibrated (CALIB_SET + CALIB_SAVE first)");
        return ESP_ERR_INVALID_STATE;
    }
    float q[MOTION_N];
    for (int i = 0; i < MOTION_N; i++) q[i] = robot_joints[i].home_rad;
    if (m->cfg.bus) {
        ESP_RETURN_ON_ERROR(board_servo_power(true), TAG, "servo power");
        vTaskDelay(pdMS_TO_TICKS(50)); /* let the servos boot */
        for (int i = 0; i < MOTION_N; i++) {
            uint16_t pos;
            esp_err_t err = servo_bus_read_position(m->cfg.bus, robot_joints[i].servo_id, &pos);
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "torque on aborted: cannot read %s (id %u): %s", robot_joints[i].name,
                         robot_joints[i].servo_id, esp_err_to_name(err));
                board_servo_power(false);
                return ESP_ERR_INVALID_RESPONSE;
            }
            q[i] = motion_calib_to_rad(&m->calib.j[i], pos);
        }
    }
    xSemaphoreTake(m->lock, portMAX_DELAY);
    file_close_locked(m);
    memcpy(m->q_cmd, q, sizeof(q));
    memcpy(m->q_hold, q, sizeof(q));
    memcpy(m->q_meas, q, sizeof(q));
    motion_limiter_reset(&m->lim, q); /* start exactly where the robot is: no jump */
    m->torque_on = true;
    m->mode = MOTION_MODE_HOLD;
    m->fault_servo_id = 0;
    xSemaphoreGive(m->lock);
    if (m->cfg.bus) return servo_bus_torque(m->cfg.bus, 0xFE, true);
    return ESP_OK;
}

esp_err_t motion_estop(motion_handle_t m)
{
    ESP_RETURN_ON_FALSE(m, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(m->lock, portMAX_DELAY);
    m->torque_on = false;
    hold_locked(m); /* -> MOTION_MODE_OFF */
    xSemaphoreGive(m->lock);
    esp_err_t err = ESP_OK;
    if (m->cfg.bus) err = servo_bus_torque(m->cfg.bus, 0xFE, false);
    esp_err_t perr = board_servo_power(false);
    ESP_LOGW(TAG, "ESTOP: torque off");
    return err != ESP_OK ? err : perr;
}

esp_err_t motion_hold(motion_handle_t m)
{
    ESP_RETURN_ON_FALSE(m, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(m->lock, portMAX_DELAY);
    hold_locked(m);
    xSemaphoreGive(m->lock);
    return ESP_OK;
}

esp_err_t motion_set_targets(motion_handle_t m, uint32_t t_robot_ms, uint16_t mask, uint16_t move_ms,
                             const float q[MOTION_N])
{
    ESP_RETURN_ON_FALSE(m && q, ESP_ERR_INVALID_ARG, TAG, "null");
    int64_t now = now_ms64();
    int64_t t_start = t_robot_ms ? robot_ms_from_u32(t_robot_ms) : now;
    if (now - t_start > CONFIG_ZANG_MOTION_STALE_MS) return ESP_ERR_INVALID_STATE; /* stale */
    if (t_start < now) t_start = now;
    xSemaphoreTake(m->lock, portMAX_DELAY);
    if (!m->torque_on) {
        xSemaphoreGive(m->lock);
        return ESP_ERR_INVALID_STATE;
    }
    file_close_locked(m);
    float goal[MOTION_N];
    memcpy(goal, m->q_cmd, sizeof(goal));
    for (int i = 0; i < MOTION_N; i++) {
        if (mask & (1u << i)) goal[i] = q[i];
    }
    motion_traj_reset(&m->traj, now, m->q_cmd, MOTION_INTERP_CUBIC);
    if (t_start > now) motion_traj_push(&m->traj, t_start, m->q_cmd); /* wait, then move */
    motion_traj_push(&m->traj, t_start + (move_ms ? move_ms : 1), goal);
    m->mode = MOTION_MODE_TARGET;
    xSemaphoreGive(m->lock);
    return ESP_OK;
}

esp_err_t motion_go_home(motion_handle_t m, uint16_t move_ms)
{
    float home[MOTION_N];
    for (int i = 0; i < MOTION_N; i++) home[i] = robot_joints[i].home_rad;
    return motion_set_targets(m, 0, (uint16_t)((1u << MOTION_N) - 1), move_ms, home);
}

esp_err_t motion_stream_start(motion_handle_t m, uint32_t stream_id, bool audio_clock)
{
    ESP_RETURN_ON_FALSE(m, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(m->lock, portMAX_DELAY);
    if (!m->torque_on) {
        xSemaphoreGive(m->lock);
        return ESP_ERR_INVALID_STATE;
    }
    file_close_locked(m);
    m->stream_id = stream_id;
    m->stream_audio_clock = audio_clock;
    m->stream_origin_ms = now_ms64();
    m->last_rx_ms = m->stream_origin_ms;
    m->stream_ending = false;
    motion_traj_reset(&m->traj, stream_now_locked(m), m->q_cmd, MOTION_INTERP_CUBIC);
    m->mode = MOTION_MODE_STREAM;
    xSemaphoreGive(m->lock);
    return ESP_OK;
}

esp_err_t motion_stream_push(motion_handle_t m, uint32_t stream_id, uint32_t t0_ms, motion_interp_t interp,
                             const uint16_t *dt_ms, const float (*q)[MOTION_N], size_t n)
{
    ESP_RETURN_ON_FALSE(m && dt_ms && q, ESP_ERR_INVALID_ARG, TAG, "null");
    esp_err_t err = ESP_OK;
    xSemaphoreTake(m->lock, portMAX_DELAY);
    if (m->mode != MOTION_MODE_STREAM || stream_id != m->stream_id) {
        err = ESP_ERR_INVALID_STATE;
    } else {
        m->last_rx_ms = now_ms64();
        m->traj.interp = interp;
        int64_t s_now = stream_now_locked(m);
        int64_t t = t0_ms;
        for (size_t k = 0; k < n; k++) {
            t += dt_ms[k];
            if (s_now - t > CONFIG_ZANG_MOTION_STALE_MS) continue; /* stale key */
            motion_traj_status_t st = motion_traj_push(&m->traj, t, q[k]);
            if (st == MOTION_TRAJ_FULL) {
                err = ESP_ERR_NO_MEM;
                break;
            }
        }
    }
    xSemaphoreGive(m->lock);
    return err;
}

esp_err_t motion_stream_end(motion_handle_t m, uint32_t stream_id)
{
    ESP_RETURN_ON_FALSE(m, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(m->lock, portMAX_DELAY);
    esp_err_t err = (m->mode == MOTION_MODE_STREAM && stream_id == m->stream_id) ? ESP_OK : ESP_ERR_INVALID_STATE;
    if (err == ESP_OK) m->stream_ending = true;
    xSemaphoreGive(m->lock);
    return err;
}

void motion_keepalive(motion_handle_t m)
{
    if (!m) return;
    xSemaphoreTake(m->lock, portMAX_DELAY);
    m->last_rx_ms = now_ms64();
    xSemaphoreGive(m->lock);
}

esp_err_t motion_play_file(motion_handle_t m, const char *path, char *audio_name, size_t audio_cap,
                           uint32_t *audio_start_ms)
{
    ESP_RETURN_ON_FALSE(m && path, ESP_ERR_INVALID_ARG, TAG, "null");
    FILE *f = fopen(path, "rb");
    ESP_RETURN_ON_FALSE(f, ESP_ERR_NOT_FOUND, TAG, "cannot open %s", path);
    uint8_t hb[ZACT_HEADER_SIZE];
    zact_header_t hdr;
    zact_status_t zs = fread(hb, 1, sizeof(hb), f) == sizeof(hb) ? zact_parse_header(hb, sizeof(hb), &hdr)
                                                                  : ZACT_ERR_SHORT;
    if (zs != ZACT_OK) {
        fclose(f);
        ESP_LOGE(TAG, "%s: bad .zact header (%d)", path, zs);
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(m->lock, portMAX_DELAY);
    if (!m->torque_on) {
        xSemaphoreGive(m->lock);
        fclose(f);
        return ESP_ERR_INVALID_STATE;
    }
    file_close_locked(m);
    m->file = f;
    m->file_hdr = hdr;
    m->file_frames_left = hdr.n_frames;
    int64_t now = now_ms64();
    m->file_origin_ms = now + CONFIG_ZANG_MOTION_FILE_LEAD_MS;
    motion_traj_reset(&m->traj, now, m->q_cmd, MOTION_INTERP_CUBIC);
    file_refill_locked(m);
    m->mode = MOTION_MODE_FILE;
    xSemaphoreGive(m->lock);
    if (audio_name && audio_cap) {
        snprintf(audio_name, audio_cap, "%s", (hdr.flags & ZACT_FLAG_AUDIO) ? hdr.audio : "");
    }
    if (audio_start_ms) *audio_start_ms = (uint32_t)(now + CONFIG_ZANG_MOTION_FILE_LEAD_MS);
    ESP_LOGI(TAG, "playing %s: %lu frames%s%s", path, (unsigned long)hdr.n_frames,
             (hdr.flags & ZACT_FLAG_AUDIO) ? ", audio " : "", (hdr.flags & ZACT_FLAG_AUDIO) ? hdr.audio : "");
    return ESP_OK;
}

esp_err_t motion_get_state(motion_handle_t m, motion_state_t *out)
{
    ESP_RETURN_ON_FALSE(m && out, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(m->lock, portMAX_DELAY);
    out->mode = m->mode;
    out->torque_on = m->torque_on;
    out->calibrated = m->calib.from_nvs;
    out->streaming = m->mode == MOTION_MODE_STREAM;
    out->stream_id = m->stream_id;
    out->stream_pos_ms = 0;
    out->stream_buffered_ms = 0;
    if (m->mode == MOTION_MODE_STREAM) {
        int64_t pos = stream_now_locked(m);
        int64_t ahead = motion_traj_end_ms(&m->traj) - pos;
        out->stream_pos_ms = pos < 0 ? 0 : (uint32_t)pos;
        out->stream_buffered_ms = ahead < 0 ? 0 : (ahead > 65535 ? 65535 : (uint32_t)ahead);
    }
    memcpy(out->q_cmd, m->q_cmd, sizeof(out->q_cmd));
    memcpy(out->q_meas, m->q_meas, sizeof(out->q_meas));
    out->fault_servo_id = m->fault_servo_id;
    out->limit_hits = m->limit_hits;
    xSemaphoreGive(m->lock);
    return ESP_OK;
}

esp_err_t motion_calib_get(motion_handle_t m, motion_calib_t *out)
{
    ESP_RETURN_ON_FALSE(m && out, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(m->lock, portMAX_DELAY);
    *out = m->calib;
    xSemaphoreGive(m->lock);
    return ESP_OK;
}

esp_err_t motion_calib_set(motion_handle_t m, uint8_t index, int16_t zero_ticks, int8_t sign, uint16_t ticks_per_rev)
{
    ESP_RETURN_ON_FALSE(m && index < MOTION_N && (sign == 1 || sign == -1), ESP_ERR_INVALID_ARG, TAG, "bad arg");
    xSemaphoreTake(m->lock, portMAX_DELAY);
    esp_err_t err = ESP_OK;
    if (m->torque_on) {
        err = ESP_ERR_INVALID_STATE; /* changing the mapping under torque would make the joint jump */
    } else {
        m->calib.j[index].zero_ticks = zero_ticks;
        m->calib.j[index].sign = sign;
        if (ticks_per_rev) m->calib.j[index].ticks_per_rev = ticks_per_rev;
    }
    xSemaphoreGive(m->lock);
    return err;
}

esp_err_t motion_calib_save(motion_handle_t m)
{
    ESP_RETURN_ON_FALSE(m, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(m->lock, portMAX_DELAY);
    motion_calib_t c = m->calib;
    xSemaphoreGive(m->lock);
    esp_err_t err = motion_calib_store(&c);
    if (err == ESP_OK) {
        xSemaphoreTake(m->lock, portMAX_DELAY);
        m->calib.from_nvs = true;
        xSemaphoreGive(m->lock);
    }
    return err;
}
