/*
 * board.h - pin table, pin validation and shared board resources (I2C bus, servo power).
 *
 * The pinout is not final: every pin is a Kconfig int, -1 = unassigned. Drivers ask board_pins_assigned()
 * before init and skip themselves (returning ESP_ERR_NOT_SUPPORTED) when a required pin is missing.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"
#include "driver/i2c_master.h"
#include "board_pins.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *group; /* "servo", "display", ... */
    const char *name;  /* Kconfig suffix: CONFIG_ZANG_PIN_<name> */
    int gpio;          /* -1 = unassigned */
} board_pin_t;

extern const board_pin_t board_pin_table[BOARD_PIN_COUNT];

/* GPIO bit mask that compiles cleanly even when the pin macro is -1 (callers still check pin >= 0). */
#define BOARD_BIT(pin) (1ULL << ((pin) < 0 ? 0 : (pin)))

/* Validate the table (duplicates, range, risky pins) and log it. ESP_ERR_INVALID_STATE on a hard conflict. */
esp_err_t board_init(void);

/* Print the pin table (assigned + unassigned) to the log. */
void board_log_pins(void);

/* True if every pin in the list is assigned (>= 0). Prefer the macro: BOARD_PINS_ASSIGNED(BOARD_PIN_A, BOARD_PIN_B). */
bool board_pins_assigned(const int *pins, size_t n);
#define BOARD_PINS_ASSIGNED(...) \
    board_pins_assigned((const int[]){__VA_ARGS__}, sizeof((const int[]){__VA_ARGS__}) / sizeof(int))

/* Shared I2C master bus (codec control, IMU). Created on first call. ESP_ERR_NOT_SUPPORTED if SDA/SCL unassigned. */
esp_err_t board_i2c_bus(i2c_master_bus_handle_t *out);

/* Servo supply switch. No-op returning ESP_OK if SERVO_PWR_EN is unassigned. */
esp_err_t board_servo_power(bool on);

/* Read the ESTOP input (true = pressed). Always false if BTN_ESTOP is unassigned. */
bool board_estop_pressed(void);

/* Battery voltage in mV from VBAT_ADC and the divider ratio; 0 if unassigned or the ADC fails. */
uint16_t board_vbat_mv(void);

#ifdef __cplusplus
}
#endif
