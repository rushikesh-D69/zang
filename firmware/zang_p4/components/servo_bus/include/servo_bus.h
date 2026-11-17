/*
 * servo_bus.h - DS serial servo bus over a UART (half duplex). Thread safe: one mutex per bus.
 * Packet encoding lives in ds_proto.h (pure C); this file only moves bytes.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct servo_bus *servo_bus_handle_t;

typedef struct {
    int uart_port;
    int tx_gpio;
    int rx_gpio;
    int dir_gpio;        /* -1 = none; else UART RTS drives the half-duplex buffer (RS485 half-duplex mode) */
    uint32_t baud;
    bool echo;           /* single-wire: TX bytes come back on RX */
    bool tx_open_drain;  /* v0.0.1 rework wiring */
    bool sync_write;     /* use SYNC WRITE for servo_bus_set_goals */
    uint32_t reply_timeout_ms;
} servo_bus_config_t;

/* Config from Kconfig + board pins. */
servo_bus_config_t servo_bus_default_config(void);

/* ESP_ERR_NOT_SUPPORTED if TX/RX pins are unassigned. */
esp_err_t servo_bus_create(const servo_bus_config_t *cfg, servo_bus_handle_t *out);
esp_err_t servo_bus_delete(servo_bus_handle_t bus);

esp_err_t servo_bus_set_goal(servo_bus_handle_t bus, uint8_t id, uint16_t position, uint16_t time_ms);
/* All goals in one cycle: one SYNC WRITE packet, or n single writes. */
esp_err_t servo_bus_set_goals(servo_bus_handle_t bus, const uint8_t *ids, const uint16_t *positions,
                              uint16_t time_ms, size_t n);
esp_err_t servo_bus_read_position(servo_bus_handle_t bus, uint8_t id, uint16_t *position);
esp_err_t servo_bus_read_voltage(servo_bus_handle_t bus, uint8_t id, uint8_t *raw);
esp_err_t servo_bus_read_temperature(servo_bus_handle_t bus, uint8_t id, uint8_t *deg_c);
/* id = DS_BROADCAST_ID (0xFE) for all servos. */
esp_err_t servo_bus_torque(servo_bus_handle_t bus, uint8_t id, bool enable);
esp_err_t servo_bus_ping(servo_bus_handle_t bus, uint8_t id);
/* Change one servo's ID (only that servo must answer old_id; never uses broadcast unlike Zing). */
esp_err_t servo_bus_set_id(servo_bus_handle_t bus, uint8_t old_id, uint8_t new_id);

typedef struct {
    uint32_t tx_packets, rx_ok, rx_timeout, rx_bad;
} servo_bus_stats_t;
void servo_bus_get_stats(servo_bus_handle_t bus, servo_bus_stats_t *out);

#ifdef __cplusplus
}
#endif
