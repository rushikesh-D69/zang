/* servo_bus.c - UART transport for the DS servo bus. */
#include "servo_bus.h"

#include <stdlib.h>
#include <string.h>
#include "board_pins.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "ds_proto.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static const char *TAG = "servo_bus";

struct servo_bus {
    servo_bus_config_t cfg;
    SemaphoreHandle_t lock;
    servo_bus_stats_t stats;
    uint8_t tx[DS_MAX_PACKET];
    uint8_t rx[DS_MAX_PACKET * 2];
};

servo_bus_config_t servo_bus_default_config(void)
{
    return (servo_bus_config_t){
        .uart_port = CONFIG_ZANG_SERVO_UART_PORT,
        .tx_gpio = BOARD_PIN_SERVO_TX,
        .rx_gpio = BOARD_PIN_SERVO_RX,
        .dir_gpio = BOARD_PIN_SERVO_DIR,
        .baud = CONFIG_ZANG_SERVO_BAUD,
#if CONFIG_ZANG_SERVO_SINGLE_WIRE_ECHO
        .echo = true,
#endif
#if CONFIG_ZANG_SERVO_SYNC_WRITE
        .sync_write = true,
#endif
#if CONFIG_ZANG_SERVO_TX_OPEN_DRAIN
        .tx_open_drain = true,
#endif
        .reply_timeout_ms = CONFIG_ZANG_SERVO_REPLY_TIMEOUT_MS,
    };
}

esp_err_t servo_bus_create(const servo_bus_config_t *cfg, servo_bus_handle_t *out)
{
    ESP_RETURN_ON_FALSE(cfg && out, ESP_ERR_INVALID_ARG, TAG, "null arg");
    if (cfg->tx_gpio < 0 || cfg->rx_gpio < 0) {
        ESP_LOGW(TAG, "SERVO_TX/SERVO_RX unassigned: servo bus disabled");
        return ESP_ERR_NOT_SUPPORTED;
    }
    struct servo_bus *bus = calloc(1, sizeof(*bus));
    ESP_RETURN_ON_FALSE(bus, ESP_ERR_NO_MEM, TAG, "no mem");
    bus->cfg = *cfg;
    bus->lock = xSemaphoreCreateMutex();
    esp_err_t err = bus->lock ? ESP_OK : ESP_ERR_NO_MEM;

    const uart_config_t ucfg = {
        .baud_rate = (int)cfg->baud,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    if (err == ESP_OK) err = uart_driver_install(cfg->uart_port, 512, 0, 0, NULL, 0);
    if (err == ESP_OK) err = uart_param_config(cfg->uart_port, &ucfg);
    if (err == ESP_OK) {
        err = uart_set_pin(cfg->uart_port, cfg->tx_gpio, cfg->rx_gpio,
                           cfg->dir_gpio >= 0 ? cfg->dir_gpio : UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    }
    if (err == ESP_OK && cfg->dir_gpio >= 0) err = uart_set_mode(cfg->uart_port, UART_MODE_RS485_HALF_DUPLEX);
    if (err == ESP_OK && cfg->tx_open_drain) err = gpio_set_direction(cfg->tx_gpio, GPIO_MODE_OUTPUT_OD); /* after uart_set_pin; keeps UART TX routing */
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "init failed: %s", esp_err_to_name(err));
        servo_bus_delete(bus);
        return err;
    }
    ESP_LOGI(TAG, "UART%d %lu baud, TX%d RX%d DIR%d, echo=%d sync_write=%d", cfg->uart_port,
             (unsigned long)cfg->baud, cfg->tx_gpio, cfg->rx_gpio, cfg->dir_gpio, cfg->echo, cfg->sync_write);
    *out = bus;
    return ESP_OK;
}

esp_err_t servo_bus_delete(servo_bus_handle_t bus)
{
    if (!bus) return ESP_ERR_INVALID_ARG;
    if (uart_is_driver_installed(bus->cfg.uart_port)) uart_driver_delete(bus->cfg.uart_port);
    if (bus->lock) vSemaphoreDelete(bus->lock);
    free(bus);
    return ESP_OK;
}

/* Caller holds the lock. Sends tx[0..n), waits until on the wire, and (single-wire) drains the echo. */
static esp_err_t send_locked(struct servo_bus *bus, int n)
{
    if (n <= 0) return ESP_ERR_INVALID_ARG;
    uart_flush_input(bus->cfg.uart_port);
    if (uart_write_bytes(bus->cfg.uart_port, bus->tx, (size_t)n) != n) return ESP_FAIL;
    esp_err_t err = uart_wait_tx_done(bus->cfg.uart_port, pdMS_TO_TICKS(50));
    bus->stats.tx_packets++;
    return err;
}

/* Caller holds the lock. Send a request built in tx[0..n) and wait for the reply of `id`. */
static esp_err_t transact_locked(struct servo_bus *bus, int n, uint8_t id, ds_reply_t *reply)
{
    ESP_RETURN_ON_ERROR(send_locked(bus, n), TAG, "send");
    const size_t skip = bus->cfg.echo ? (size_t)n : 0;
    size_t got = 0;
    TickType_t deadline = xTaskGetTickCount() + pdMS_TO_TICKS(bus->cfg.reply_timeout_ms) + 1;
    while (got < sizeof(bus->rx)) {
        TickType_t now = xTaskGetTickCount();
        if ((int32_t)(deadline - now) <= 0) break;
        int r = uart_read_bytes(bus->cfg.uart_port, &bus->rx[got], sizeof(bus->rx) - got, deadline - now);
        if (r <= 0) break;
        got += (size_t)r;
        ds_status_t st = ds_find_reply(bus->rx, got, skip, id, reply);
        if (st == DS_OK) {
            bus->stats.rx_ok++;
            return reply->error ? ESP_ERR_INVALID_RESPONSE : ESP_OK;
        }
        if (st == DS_ERR_CHECKSUM || st == DS_ERR_LENGTH) bus->stats.rx_bad++;
    }
    bus->stats.rx_timeout++;
    return ESP_ERR_TIMEOUT;
}

#define LOCKED(bus, expr)                                                                                        \
    ({                                                                                                           \
        xSemaphoreTake((bus)->lock, portMAX_DELAY);                                                              \
        esp_err_t _e = (expr);                                                                                   \
        xSemaphoreGive((bus)->lock);                                                                             \
        _e;                                                                                                      \
    })

esp_err_t servo_bus_set_goal(servo_bus_handle_t bus, uint8_t id, uint16_t position, uint16_t time_ms)
{
    ESP_RETURN_ON_FALSE(bus, ESP_ERR_INVALID_ARG, TAG, "null bus");
    return LOCKED(bus, send_locked(bus, ds_build_goal(bus->tx, sizeof(bus->tx), id, position, time_ms)));
}

esp_err_t servo_bus_set_goals(servo_bus_handle_t bus, const uint8_t *ids, const uint16_t *positions,
                              uint16_t time_ms, size_t n)
{
    ESP_RETURN_ON_FALSE(bus && ids && positions && n, ESP_ERR_INVALID_ARG, TAG, "bad arg");
    xSemaphoreTake(bus->lock, portMAX_DELAY);
    esp_err_t err = ESP_OK;
    if (bus->cfg.sync_write) {
        err = send_locked(bus, ds_build_sync_goal(bus->tx, sizeof(bus->tx), ids, positions, time_ms, n));
    } else {
        for (size_t i = 0; i < n && err == ESP_OK; i++) {
            err = send_locked(bus, ds_build_goal(bus->tx, sizeof(bus->tx), ids[i], positions[i], time_ms));
        }
    }
    xSemaphoreGive(bus->lock);
    return err;
}

static esp_err_t read_reg(struct servo_bus *bus, uint8_t id, uint8_t reg, uint8_t nbytes, uint8_t *out)
{
    ESP_RETURN_ON_FALSE(bus && out && id != DS_BROADCAST_ID, ESP_ERR_INVALID_ARG, TAG, "bad arg");
    xSemaphoreTake(bus->lock, portMAX_DELAY);
    ds_reply_t r;
    esp_err_t err = transact_locked(bus, ds_build_read(bus->tx, sizeof(bus->tx), id, reg, nbytes), id, &r);
    if (err == ESP_OK && r.n_params < nbytes) err = ESP_ERR_INVALID_SIZE;
    if (err == ESP_OK) memcpy(out, r.params, nbytes);
    xSemaphoreGive(bus->lock);
    return err;
}

esp_err_t servo_bus_read_position(servo_bus_handle_t bus, uint8_t id, uint16_t *position)
{
    uint8_t b[2];
    esp_err_t err = read_reg(bus, id, DS_REG_PRESENT_POSITION, 2, b);
    if (err == ESP_OK) *position = ds_be16(b);
    return err;
}

esp_err_t servo_bus_read_voltage(servo_bus_handle_t bus, uint8_t id, uint8_t *raw)
{
    return read_reg(bus, id, DS_REG_PRESENT_VOLTAGE, 1, raw);
}

esp_err_t servo_bus_read_temperature(servo_bus_handle_t bus, uint8_t id, uint8_t *deg_c)
{
    return read_reg(bus, id, DS_REG_PRESENT_TEMP, 1, deg_c);
}

esp_err_t servo_bus_torque(servo_bus_handle_t bus, uint8_t id, bool enable)
{
    ESP_RETURN_ON_FALSE(bus, ESP_ERR_INVALID_ARG, TAG, "null bus");
    return LOCKED(bus, send_locked(bus, ds_build_torque(bus->tx, sizeof(bus->tx), id, enable)));
}

esp_err_t servo_bus_ping(servo_bus_handle_t bus, uint8_t id)
{
    ESP_RETURN_ON_FALSE(bus, ESP_ERR_INVALID_ARG, TAG, "null bus");
    ds_reply_t r;
    return LOCKED(bus, transact_locked(bus, ds_build(bus->tx, sizeof(bus->tx), id, DS_INST_PING, NULL, 0), id, &r));
}

esp_err_t servo_bus_set_id(servo_bus_handle_t bus, uint8_t old_id, uint8_t new_id)
{
    ESP_RETURN_ON_FALSE(bus && old_id != DS_BROADCAST_ID && new_id < DS_BROADCAST_ID, ESP_ERR_INVALID_ARG, TAG,
                        "bad id");
    return LOCKED(bus, send_locked(bus, ds_build_write(bus->tx, sizeof(bus->tx), old_id, DS_REG_ID, &new_id, 1)));
}

void servo_bus_get_stats(servo_bus_handle_t bus, servo_bus_stats_t *out)
{
    if (!bus || !out) return;
    xSemaphoreTake(bus->lock, portMAX_DELAY);
    *out = bus->stats;
    xSemaphoreGive(bus->lock);
}
