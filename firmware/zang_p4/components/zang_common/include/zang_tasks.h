/*
 * zang_tasks.h - the ONE place that sets FreeRTOS task names, stacks, priorities and cores.
 * Rule: no xTaskCreate* call in the firmware may use literal numbers; use ZANG_TASK_* from here.
 *
 * Core 1 is reserved for hard-real-time motion (servo loop, IMU). Core 0 runs networking, audio, UI.
 * Priorities: higher = more urgent. ESP-Hosted / lwIP tasks run at ~18-23 on core 0.
 */
#pragma once

#include "freertos/FreeRTOS.h"

typedef struct {
    const char *name;
    uint32_t stack_bytes;
    UBaseType_t priority;
    BaseType_t core; /* 0, 1, or tskNO_AFFINITY */
} zang_task_def_t;

/*                                         name            stack  prio  core */
#define ZANG_TASK_MOTION     ((zang_task_def_t){"motion",     6144, 22, 1})
#define ZANG_TASK_IMU        ((zang_task_def_t){"imu",        4096, 21, 1})
#define ZANG_TASK_AUDIO_OUT  ((zang_task_def_t){"audio_out",  6144, 19, 0})
#define ZANG_TASK_AUDIO_IN   ((zang_task_def_t){"audio_in",   4096, 18, 0})
#define ZANG_TASK_ZLINK_UDP  ((zang_task_def_t){"zlink_udp",  6144, 16, 0})
#define ZANG_TASK_ZLINK_TCP  ((zang_task_def_t){"zlink_tcp",  6144, 12, 0})
#define ZANG_TASK_STATE_TX   ((zang_task_def_t){"state_tx",   4096, 14, 0})
#define ZANG_TASK_APP_EVENTS ((zang_task_def_t){"app_evt",    4096, 10, 0})
#define ZANG_TASK_CAMERA     ((zang_task_def_t){"camera",     6144,  8, 0})
#define ZANG_TASK_DISPLAY    ((zang_task_def_t){"display",    6144,  6, 0})

/* Create a task from a table entry. Returns pdPASS like xTaskCreatePinnedToCore. */
static inline BaseType_t zang_task_create(zang_task_def_t def, TaskFunction_t fn, void *arg, TaskHandle_t *out)
{
    return xTaskCreatePinnedToCore(fn, def.name, def.stack_bytes, arg, def.priority, out, def.core);
}
