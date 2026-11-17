/*
 * motion.h - the real-time motion runtime (one task on core 1).
 *
 * Every cycle: pick the target from the active mode (HOLD / TARGET / STREAM / FILE), pass it through the
 * last-line limiter (robot_joints.h limits), convert to servo ticks with the NVS calibration and write the bus.
 * All angles are MODEL joint angles in radians (ai_stack/config/robot.json frame).
 * Thread safe: every function may be called from any task.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"
#include "motion_math.h"
#include "servo_bus.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MOTION_MODE_OFF = 0, /* torque off (boot default, ESTOP) */
    MOTION_MODE_HOLD,
    MOTION_MODE_TARGET,  /* JOINT_TARGETS / GO_HOME */
    MOTION_MODE_STREAM,  /* TRAJ_CHUNK stream */
    MOTION_MODE_FILE,    /* .zact playback */
} motion_mode_t;

typedef enum {
    MOTION_EV_TARGET_DONE,
    MOTION_EV_STREAM_DONE,  /* STREAM_END received and all keys played */
    MOTION_EV_FILE_DONE,
    MOTION_EV_WATCHDOG,     /* no host frame for CONFIG_ZANG_MOTION_WATCHDOG_MS while streaming -> HOLD */
    MOTION_EV_FAULT,        /* servo stopped answering (see motion_state_t.fault_servo_id) */
    MOTION_EV_ESTOP,        /* hardware ESTOP input pressed: torque already off */
} motion_event_t;

typedef void (*motion_event_cb_t)(motion_event_t ev, void *ctx);
/* Optional audio clock for ZLINK clock = 1 streams. Return false if no audio is playing. */
typedef bool (*motion_audio_clock_t)(void *ctx, uint32_t *ms);

typedef struct {
    servo_bus_handle_t bus;          /* NULL = run the loop without hardware (twin-only / bring-up) */
    motion_event_cb_t on_event;
    motion_audio_clock_t audio_clock;
    void *ctx;
} motion_config_t;

typedef struct {
    motion_mode_t mode;
    bool torque_on;
    bool calibrated;
    bool streaming;
    uint32_t stream_id;
    uint32_t stream_pos_ms;      /* playhead on the stream clock (0 if no stream) */
    uint32_t stream_buffered_ms; /* trajectory queued ahead of the playhead */
    float q_cmd[MOTION_N];  /* after the limiter */
    float q_meas[MOTION_N]; /* NAN = not read yet */
    uint8_t fault_servo_id; /* 0 = none */
    uint32_t limit_hits;    /* targets clamped to position limits since boot */
} motion_state_t;

typedef struct motion *motion_handle_t;

esp_err_t motion_start(const motion_config_t *cfg, motion_handle_t *out);

/* Robot clock used by ZLink t_robot_ms (ms since boot, wraps after 49 days). */
uint32_t motion_now_ms(void);

esp_err_t motion_torque_on(motion_handle_t m);  /* reads the pose, holds it, enables torque + servo power */
esp_err_t motion_estop(motion_handle_t m);      /* torque off + servo power off, immediately */
esp_err_t motion_hold(motion_handle_t m);       /* stop any stream/file, hold the commanded pose */
esp_err_t motion_go_home(motion_handle_t m, uint16_t move_ms);
/* Direct target: joints in mask move to q over move_ms, starting at t_robot_ms (0 = now). Unmasked hold. */
esp_err_t motion_set_targets(motion_handle_t m, uint32_t t_robot_ms, uint16_t mask, uint16_t move_ms,
                             const float q[MOTION_N]);

esp_err_t motion_stream_start(motion_handle_t m, uint32_t stream_id, bool audio_clock);
/* Keys at stream times t0_ms + cumulative dt_ms. Stale keys (older than STALE_MS) are dropped. */
esp_err_t motion_stream_push(motion_handle_t m, uint32_t stream_id, uint32_t t0_ms, motion_interp_t interp,
                             const uint16_t *dt_ms, const float (*q)[MOTION_N], size_t n);
esp_err_t motion_stream_end(motion_handle_t m, uint32_t stream_id);
/* Any frame from the host feeds the watchdog. */
void motion_keepalive(motion_handle_t m);

/* Play a .zact file. *audio_name (optional) receives the companion audio file name ("" if none) and
 * *audio_start_ms the robot time at which frame 0 plays, so the app can start the audio in sync. */
esp_err_t motion_play_file(motion_handle_t m, const char *path, char *audio_name, size_t audio_cap,
                           uint32_t *audio_start_ms);

esp_err_t motion_get_state(motion_handle_t m, motion_state_t *out);

/* ticks_per_rev 0 = leave unchanged (ZLink CALIB_SET's optional trailing u16). */
esp_err_t motion_calib_set(motion_handle_t m, uint8_t index, int16_t zero_ticks, int8_t sign, uint16_t ticks_per_rev);
/* Current table (what CALIB_REPORT sends). */
esp_err_t motion_calib_get(motion_handle_t m, motion_calib_t *out);
esp_err_t motion_calib_save(motion_handle_t m);

/* NVS calibration storage (motion_calib_nvs.c). */
esp_err_t motion_calib_load(motion_calib_t *c);
esp_err_t motion_calib_store(const motion_calib_t *c);

#ifdef __cplusplus
}
#endif
