/* imu.c - QMI8658 driver + complementary filter. Compiled to a stub unless CONFIG_ZANG_IMU_ENABLE. */
#include "imu.h"

#include "esp_log.h"
#include "sdkconfig.h"

static const char *TAG = "imu";

#if !CONFIG_ZANG_IMU_ENABLE

esp_err_t imu_init(imu_handle_t *out)
{
    ESP_LOGI(TAG, "no IMU configured (board v0.0.1): STATE reports rpy = n/a");
    return ESP_ERR_NOT_SUPPORTED;
}
esp_err_t imu_get(imu_handle_t imu, imu_state_t *out) { return ESP_ERR_NOT_SUPPORTED; }

#else

#include <math.h>
#include <stdlib.h>
#include "board.h"
#include "esp_check.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "zang_tasks.h"

#define REG_WHO_AM_I 0x00
#define REG_CTRL1    0x02
#define REG_CTRL2    0x03
#define REG_CTRL3    0x04
#define REG_CTRL7    0x08
#define REG_AX_L     0x35
#define WHO_AM_I_VAL 0x05
#define ACC_LSB_G    4096.0f /* +-8 g */
#define GYR_LSB_DPS  64.0f   /* +-512 dps */
#define ALPHA        0.98f   /* complementary filter: gyro weight */

struct imu {
    i2c_master_dev_handle_t dev;
    SemaphoreHandle_t lock;
    imu_state_t st;
};

static esp_err_t wr(struct imu *m, uint8_t reg, uint8_t v)
{
    uint8_t b[2] = {reg, v};
    return i2c_master_transmit(m->dev, b, 2, 20);
}

static esp_err_t rd(struct imu *m, uint8_t reg, uint8_t *buf, size_t n)
{
    return i2c_master_transmit_receive(m->dev, &reg, 1, buf, n, 20);
}

static void imu_task(void *arg)
{
    struct imu *m = arg;
    const float dt = 1.0f / CONFIG_ZANG_IMU_RATE_HZ;
    TickType_t wake = xTaskGetTickCount();
    imu_state_t s = {0};
    for (;;) {
        vTaskDelayUntil(&wake, pdMS_TO_TICKS(1000 / CONFIG_ZANG_IMU_RATE_HZ));
        uint8_t b[12];
        if (rd(m, REG_AX_L, b, sizeof(b)) != ESP_OK) continue;
        for (int i = 0; i < 3; i++) {
            s.acc_g[i] = (int16_t)(b[2 * i] | (b[2 * i + 1] << 8)) / ACC_LSB_G;
            s.gyro_dps[i] = (int16_t)(b[6 + 2 * i] | (b[7 + 2 * i] << 8)) / GYR_LSB_DPS;
        }
        float acc_roll = atan2f(s.acc_g[1], s.acc_g[2]);
        float acc_pitch = atan2f(-s.acc_g[0], sqrtf(s.acc_g[1] * s.acc_g[1] + s.acc_g[2] * s.acc_g[2]));
        const float d2r = 0.0174532925f;
        if (!s.valid) {
            s.rpy_rad[0] = acc_roll;
            s.rpy_rad[1] = acc_pitch;
            s.valid = true;
        } else {
            s.rpy_rad[0] = ALPHA * (s.rpy_rad[0] + s.gyro_dps[0] * d2r * dt) + (1 - ALPHA) * acc_roll;
            s.rpy_rad[1] = ALPHA * (s.rpy_rad[1] + s.gyro_dps[1] * d2r * dt) + (1 - ALPHA) * acc_pitch;
        }
        s.rpy_rad[2] += s.gyro_dps[2] * d2r * dt;
        xSemaphoreTake(m->lock, portMAX_DELAY);
        m->st = s;
        xSemaphoreGive(m->lock);
    }
}

esp_err_t imu_init(imu_handle_t *out)
{
    ESP_RETURN_ON_FALSE(out, ESP_ERR_INVALID_ARG, TAG, "null");
    i2c_master_bus_handle_t bus;
    ESP_RETURN_ON_ERROR(board_i2c_bus(&bus), TAG, "i2c");
    struct imu *m = calloc(1, sizeof(*m));
    ESP_RETURN_ON_FALSE(m, ESP_ERR_NO_MEM, TAG, "no mem");
    m->lock = xSemaphoreCreateMutex();
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = CONFIG_ZANG_IMU_I2C_ADDR,
                              .scl_speed_hz = CONFIG_ZANG_I2C_FREQ_HZ};
    ESP_RETURN_ON_ERROR(i2c_master_bus_add_device(bus, &dc, &m->dev), TAG, "add device");
    uint8_t who = 0;
    if (rd(m, REG_WHO_AM_I, &who, 1) != ESP_OK || who != WHO_AM_I_VAL) {
        ESP_LOGW(TAG, "QMI8658 not found at 0x%02x (who=0x%02x)", CONFIG_ZANG_IMU_I2C_ADDR, who);
        i2c_master_bus_rm_device(m->dev);
        free(m);
        return ESP_ERR_NOT_FOUND;
    }
    ESP_RETURN_ON_ERROR(wr(m, REG_CTRL1, 0x40), TAG, "ctrl1");  /* address auto-increment, little endian */
    ESP_RETURN_ON_ERROR(wr(m, REG_CTRL2, 0x23), TAG, "ctrl2");  /* acc +-8 g, ODR 1 kHz */
    ESP_RETURN_ON_ERROR(wr(m, REG_CTRL3, 0x53), TAG, "ctrl3");  /* gyro +-512 dps, ODR 1 kHz */
    ESP_RETURN_ON_ERROR(wr(m, REG_CTRL7, 0x03), TAG, "ctrl7");  /* enable acc + gyro */
    if (zang_task_create(ZANG_TASK_IMU, imu_task, m, NULL) != pdPASS) return ESP_ERR_NO_MEM;
    ESP_LOGI(TAG, "QMI8658 at 0x%02x, %d Hz", CONFIG_ZANG_IMU_I2C_ADDR, CONFIG_ZANG_IMU_RATE_HZ);
    *out = m;
    return ESP_OK;
}

esp_err_t imu_get(imu_handle_t m, imu_state_t *out)
{
    ESP_RETURN_ON_FALSE(m && out, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(m->lock, portMAX_DELAY);
    *out = m->st;
    xSemaphoreGive(m->lock);
    return ESP_OK;
}

#endif
