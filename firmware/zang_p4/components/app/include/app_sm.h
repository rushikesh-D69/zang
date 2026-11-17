/*
 * app_sm.h - top-level robot state machine (pure C, host-tested).
 *
 *   BOOT --boot_done--> IDLE (torque off) --torque_on--> HOLD <--> STREAMING / PLAYING_FILE
 *   any --estop--> ESTOP (latched) --torque_on--> HOLD
 * Commands that would move the robot are only accepted in HOLD / STREAMING / PLAYING_FILE.
 */
#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    APP_BOOT = 0,
    APP_IDLE,
    APP_HOLD,
    APP_STREAMING,
    APP_PLAYING_FILE,
    APP_ESTOP,
    APP_STATE_COUNT
} app_state_t;

typedef enum {
    APP_EV_BOOT_DONE = 0,
    APP_EV_TORQUE_ON,    /* torque enabled successfully */
    APP_EV_HOLD,         /* host HOLD or a target move */
    APP_EV_ESTOP,        /* host ESTOP, hardware ESTOP, or torque-on failure */
    APP_EV_STREAM_START,
    APP_EV_STREAM_DONE,
    APP_EV_PLAY_FILE,
    APP_EV_FILE_DONE,
    APP_EV_WATCHDOG,
    APP_EV_COUNT
} app_event_t;

/* Apply an event. Returns true if accepted (state may be unchanged), false if not allowed in this state. */
bool app_sm_step(app_state_t *state, app_event_t ev);
/* True if commands that move the robot are allowed. */
bool app_sm_can_move(app_state_t s);
const char *app_state_name(app_state_t s);

#ifdef __cplusplus
}
#endif
