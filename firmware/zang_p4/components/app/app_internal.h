/* app_internal.h - shared between app.c (dispatcher, STATE) and app_media.c (display, camera, microphones). */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "app.h"
#include "esp_err.h"
#include "zlink.h"
#include "zlink_server.h"

/* Provided by app.c */
void app_send(zlink_via_t via, uint8_t type, const uint8_t *payload, uint16_t len);
void app_send_event(uint8_t kind, const uint8_t *data, uint16_t n);
bool app_has_peer(void);
uint8_t app_err_to_status(esp_err_t e);
const app_ctx_t *app_ctx(void);
/* A host face message (FACE_SET/PARAMS/TRACK) arrived: the state machine stops choosing faces (except ESTOP). */
bool app_host_owns_face(void);

/* Provided by app_media.c */
void app_media_start(const app_ctx_t *ctx);
/* Handles FACE_* and DISPLAY_* messages. Returns true if the frame type was one of them. */
bool app_media_on_frame(const zlink_frame_t *f, zlink_via_t via);
/* CMD MIC_STREAM / CAMERA_STREAM / SET_VOLUME / SET_BRIGHTNESS / SNAPSHOT. Returns ESP_ERR_NOT_FOUND for any other cmd. */
esp_err_t app_media_cmd(const zlink_cmd_t *cmd);
/* Display mode and emotion for the STATE tail. */
uint8_t app_media_display_mode(void);
uint8_t app_media_emotion(void);
