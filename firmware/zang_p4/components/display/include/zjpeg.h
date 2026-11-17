/*
 * zjpeg.h - JPEG -> RGB565 on the ESP32-P4 hardware decoder (driver/jpeg_decode.h), for DISPLAY_IMAGE and the
 * camera preview. Baseline JPEG only (the hardware does not decode progressive files).
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "canvas.h"
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Decode jpg and scale it into dst (dw x dh pixels, panel byte order, stride dw). Thread safe (one decode at a
 * time). ESP_ERR_INVALID_ARG for a stream the decoder cannot parse or larger than 1280x720. */
esp_err_t zjpeg_decode_to_panel(const uint8_t *jpg, size_t len, img_fit_t mode, uint16_t bg, uint16_t *dst, int dw,
                                int dh);
/* Allocator for DISPLAY_IMAGE reassembly buffers (cache-line aligned PSRAM) and its release. */
void *zjpeg_alloc_input(size_t n);
void zjpeg_free(void *p);

#ifdef __cplusplus
}
#endif
