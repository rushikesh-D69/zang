/*
 * camera.h - DVP camera (OV2640 via esp_cam_sensor) on the P4 DVP controller.
 * Frames arrive on the camera task; the callback must return quickly (copy or hand off).
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint8_t *data;
    size_t len;      /* bytes received (JPEG: actual JPEG size) */
    int width, height;
    bool jpeg;       /* always true: the sensor compresses (OV2640 JPEG) */
    int64_t t_us;    /* capture-complete time (esp_timer) */
    uint32_t seq;
} camera_frame_t;

typedef enum { CAMERA_SIZE_QVGA = 0, CAMERA_SIZE_VGA = 1, CAMERA_SIZE_COUNT } camera_size_t; /* 320x240, 640x480 */

typedef void (*camera_frame_cb_t)(const camera_frame_t *frame, void *ctx);

typedef struct camera *camera_handle_t;

/* ESP_ERR_NOT_SUPPORTED if disabled in Kconfig or pins unassigned; ESP_ERR_NOT_FOUND if no sensor answers. */
esp_err_t camera_init(camera_handle_t *out);
/* Resolution and JPEG quality (1..100). A new size rebuilds the DVP controller (about 100 ms; frames are lost). */
esp_err_t camera_configure(camera_handle_t cam, camera_size_t size, uint8_t quality);
/* Start the sensor stream: cb runs on the camera task for every JPEG frame (the buffer is only valid inside it). */
esp_err_t camera_start(camera_handle_t cam, camera_frame_cb_t cb, void *ctx);
esp_err_t camera_stop(camera_handle_t cam);

typedef struct {
    uint32_t frames, dropped;
    float fps;
    const char *sensor; /* e.g. "OV2640" */
    const char *format;
} camera_stats_t;
esp_err_t camera_get_stats(camera_handle_t cam, camera_stats_t *out);

#ifdef __cplusplus
}
#endif
