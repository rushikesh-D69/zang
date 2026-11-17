/*
 * app.h - wires drivers to ZLink: dispatches host messages, runs the state machine, publishes STATE.
 * Any handle may be NULL (driver absent or skipped): the app degrades instead of failing.
 */
#pragma once

#include "audio.h"
#include "camera.h"
#include "display.h"
#include "esp_err.h"
#include "imu.h"
#include "motion.h"

#ifdef __cplusplus
extern "C" {
#endif

/* EVENT (0x60) kinds sent by this firmware. ZLINK.md lists examples but no codes; these are the firmware's. */
typedef enum {
    APP_EVENT_STATE_CHANGED = 0x01, /* data: u8 new app_state_t */
    APP_EVENT_STREAM_DONE = 0x02,
    APP_EVENT_FILE_DONE = 0x03,
    APP_EVENT_WATCHDOG = 0x04,
    APP_EVENT_SERVO_FAULT = 0x05,   /* data: u8 servo id */
    APP_EVENT_ESTOP = 0x06,
    APP_EVENT_TARGET_DONE = 0x07,
} app_event_kind_t;

typedef struct {
    motion_handle_t motion;
    audio_handle_t audio;
    display_handle_t display;
    camera_handle_t camera;
    imu_handle_t imu;
    bool board_ok; /* false = pin conflict: stay in ESTOP */
} app_ctx_t;

/* Before motion_start(): route motion events and the audio clock through the app. */
void app_motion_config(motion_config_t *cfg);
/* Before motion_start(): the audio handle used by the audio clock (may be NULL). */
void app_set_audio(audio_handle_t audio);

/* Starts the ZLink servers and the STATE publisher. Call once, after the drivers. */
esp_err_t app_start(const app_ctx_t *ctx);

#ifdef __cplusplus
}
#endif
