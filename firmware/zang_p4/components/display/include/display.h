/*
 * display.h - SPI TFT: face (SDF renderer + animation engine), images, text, camera preview, status.
 * A PSRAM framebuffer (panel byte order) is drawn by the display task at CONFIG_ZANG_LCD_FPS. Only the changed
 * rectangle is rendered and sent, through two internal-RAM DMA line buffers.
 * Modes (ZLink DISPLAY_MODE): 0 face, 1 image, 2 text, 3 camera preview, 4 status.
 * Thread safe: every call may come from any task.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"
#include "face_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DISPLAY_IMAGE_DIR "/sdcard/img" /* stored images: DISPLAY_IMAGE_DIR/<id>.jpg (SD mount point + /img) */

typedef enum {
    DISPLAY_MODE_FACE = 0,
    DISPLAY_MODE_IMAGE = 1,
    DISPLAY_MODE_TEXT = 2,
    DISPLAY_MODE_CAMERA = 3,
    DISPLAY_MODE_STATUS = 4,
} display_mode_t;

/* Legacy coarse faces used by the app state machine; each maps to a faces.json emotion. */
typedef enum {
    DISPLAY_FACE_NEUTRAL = 0,
    DISPLAY_FACE_HAPPY,
    DISPLAY_FACE_SAD,
    DISPLAY_FACE_SURPRISED,
    DISPLAY_FACE_SLEEPY,
    DISPLAY_FACE_LISTENING,
    DISPLAY_FACE_THINKING,
    DISPLAY_FACE_ERROR,
    DISPLAY_FACE_COUNT
} display_face_t;

typedef struct {
    float full_fps;    /* full-screen frames per second */
    float partial_fps; /* face-region renders + updates per second */
} display_fps_t;

typedef struct display *display_handle_t;

/* Playhead of the current motion stream, for FACE_TRACK. Return false when no stream plays. */
typedef bool (*display_stream_fn_t)(void *ctx, uint32_t *stream_id, uint32_t *pos_ms);

/* ESP_ERR_NOT_SUPPORTED if the LCD pins are unassigned. Starts the display task. */
esp_err_t display_init(display_handle_t *out);
esp_err_t display_set_backlight(display_handle_t d, uint8_t percent);
void display_set_stream_source(display_handle_t d, display_stream_fn_t fn, void *ctx);

/* ---- face (mode 0). FACE_SET / FACE_PARAMS / FACE_TRACK also switch to face mode. ---- */
esp_err_t display_face_set(display_handle_t d, uint8_t emotion, uint8_t intensity, uint16_t blend_ms, int8_t gaze_x,
                           int8_t gaze_y, uint8_t flags);
esp_err_t display_face_params(display_handle_t d, uint16_t blend_ms, int n, const uint8_t *ids, const float *values);
esp_err_t display_face_track(display_handle_t d, uint32_t stream_id, const face_key_t *keys, int n);
esp_err_t display_set_face(display_handle_t d, display_face_t face); /* legacy */

/* ---- other modes ---- */
/* Mode 1 shows image_id: the cached last decoded image, else /sdcard/img/<id>.jpg (ESP_ERR_NOT_FOUND). */
esp_err_t display_set_mode(display_handle_t d, display_mode_t mode, uint16_t image_id);
display_mode_t display_get_mode(display_handle_t d);
uint8_t display_get_emotion(display_handle_t d);
/* Text (mode 2): size 1..4, RGB565 colours, duration 0 = until the mode changes, else back to the face after it. */
esp_err_t display_show_text(display_handle_t d, uint8_t size, uint16_t fg, uint16_t bg, uint16_t duration_ms,
                            const char *text, int n);
/* Decode a JPEG (RGB565 240x240, fit) and show it as image_id (mode 1). */
esp_err_t display_show_jpeg(display_handle_t d, uint16_t image_id, const uint8_t *jpg, size_t len);
/* Same for raw RGB565 little-endian 240x240 (DISPLAY_IMAGE format 0). */
esp_err_t display_show_rgb565(display_handle_t d, uint16_t image_id, const uint16_t *px, size_t n_px);
/* One status text for mode 4 (and shown as-is: ASCII, wraps). */
esp_err_t display_set_status(display_handle_t d, const char *text);
/* Camera preview frame (JPEG); drawn only while the mode is 3 (cover-scaled to the panel). */
esp_err_t display_camera_frame(display_handle_t d, const uint8_t *jpg, size_t len);
/* Blit RGB565 pixels (native little-endian uint16) into the framebuffer. */
esp_err_t display_blit_rgb565(display_handle_t d, int x, int y, int w, int h, const uint16_t *px);
/* Store a JPEG as DISPLAY_IMAGE_DIR/<id>.jpg (creates the folder). */
esp_err_t display_store_jpeg(uint16_t image_id, const uint8_t *jpg, size_t len);
/* Measure FPS (blocks ~1 s, pauses the animator). */
esp_err_t display_self_test(display_handle_t d, display_fps_t *out);

#ifdef __cplusplus
}
#endif
