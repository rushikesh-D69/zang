/*
 * zlink_server.h - ZLink transports on the robot: UDP (real-time, port 4210) and TCP (bulk, port 4211).
 * One TCP client at a time. The UDP peer is whoever sent the last valid frame; STATE goes back to it.
 * Received frames are delivered to on_frame from the transport task, so keep the callback short.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"
#include "zlink.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { ZLINK_VIA_UDP = 0, ZLINK_VIA_TCP = 1 } zlink_via_t;

typedef void (*zlink_on_frame_t)(const zlink_frame_t *frame, zlink_via_t via, void *ctx);

typedef struct {
    uint16_t udp_port;
    uint16_t tcp_port;
    zlink_on_frame_t on_frame;
    void *ctx;
} zlink_server_config_t;

typedef struct zlink_server *zlink_server_handle_t;

esp_err_t zlink_server_start(const zlink_server_config_t *cfg, zlink_server_handle_t *out);
/* Encode + send a message to the current peer on `via`. ESP_ERR_INVALID_STATE if no peer yet. Thread safe. */
esp_err_t zlink_server_send(zlink_server_handle_t s, zlink_via_t via, uint8_t type, const uint8_t *payload,
                            uint16_t len);
bool zlink_server_has_peer(zlink_server_handle_t s);

typedef struct {
    uint32_t rx_frames, rx_bad, tx_frames, tx_fail, tcp_clients;
} zlink_server_stats_t;
void zlink_server_get_stats(zlink_server_handle_t s, zlink_server_stats_t *out);

#ifdef __cplusplus
}
#endif
