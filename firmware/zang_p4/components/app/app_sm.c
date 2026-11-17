/* app_sm.c - table-driven state machine. Pure C. */
#include "app_sm.h"

#define X APP_STATE_COUNT /* "not allowed" */

/* next[state][event] */
static const unsigned char next[APP_STATE_COUNT][APP_EV_COUNT] = {
    /*                 BOOT_DONE  TORQUE_ON  HOLD      ESTOP      STREAM_ST      STREAM_DONE FILE              FILE_DONE WATCHDOG */
    [APP_BOOT]      = {APP_IDLE,  X,         X,        APP_ESTOP, X,             X,          X,                X,        X},
    [APP_IDLE]      = {X,         APP_HOLD,  X,        APP_ESTOP, X,             X,          X,                X,        X},
    [APP_HOLD]      = {X,         APP_HOLD,  APP_HOLD, APP_ESTOP, APP_STREAMING, APP_HOLD,   APP_PLAYING_FILE, APP_HOLD, APP_HOLD},
    [APP_STREAMING] = {X,         APP_HOLD,  APP_HOLD, APP_ESTOP, APP_STREAMING, APP_HOLD,   APP_PLAYING_FILE, APP_STREAMING, APP_HOLD},
    [APP_PLAYING_FILE] = {X,      APP_HOLD,  APP_HOLD, APP_ESTOP, APP_STREAMING, APP_PLAYING_FILE, APP_PLAYING_FILE, APP_HOLD, APP_PLAYING_FILE},
    [APP_ESTOP]     = {X,         APP_HOLD,  X,        APP_ESTOP, X,             X,          X,                X,        X},
};

bool app_sm_step(app_state_t *s, app_event_t ev)
{
    if (!s || *s >= APP_STATE_COUNT || ev >= APP_EV_COUNT) return false;
    unsigned char n = next[*s][ev];
    if (n == X) return false;
    *s = (app_state_t)n;
    return true;
}

bool app_sm_can_move(app_state_t s) { return s == APP_HOLD || s == APP_STREAMING || s == APP_PLAYING_FILE; }

const char *app_state_name(app_state_t s)
{
    static const char *names[] = {"BOOT", "IDLE", "HOLD", "STREAMING", "PLAYING_FILE", "ESTOP"};
    return s < APP_STATE_COUNT ? names[s] : "?";
}
