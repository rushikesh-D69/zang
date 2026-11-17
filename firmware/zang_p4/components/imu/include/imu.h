/* imu.h - optional QMI8658 6-axis IMU with a complementary attitude filter (roll/pitch; yaw = gyro integral). */
#pragma once

#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float rpy_rad[3];   /* roll, pitch, yaw (yaw drifts: no magnetometer) */
    float gyro_dps[3];
    float acc_g[3];
    bool valid;
} imu_state_t;

typedef struct imu *imu_handle_t;

/* ESP_ERR_NOT_SUPPORTED when disabled in Kconfig (default on board v0.0.1) or I2C is unassigned. */
esp_err_t imu_init(imu_handle_t *out);
esp_err_t imu_get(imu_handle_t imu, imu_state_t *out);

#ifdef __cplusplus
}
#endif
