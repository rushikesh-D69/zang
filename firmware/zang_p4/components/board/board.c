/* board.c - pin validation/report and shared board resources. */
#include "board.h"

#include <string.h>
#include "driver/gpio.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static const char *TAG = "board";

#define P4_GPIO_MAX 54

static i2c_master_bus_handle_t s_i2c_bus;
static SemaphoreHandle_t s_i2c_lock;
static bool s_servo_pwr_ready;

static bool pin_is_risky(int gpio, const char **why)
{
    if (gpio >= 24 && gpio <= 27) {
        *why = "USB pin (USB-Serial-JTAG / FS OTG)";
        return true;
    }
    if (gpio >= 34 && gpio <= 38) {
        *why = "strapping pin (boot mode / UART0)";
        return true;
    }
    return false;
}

void board_log_pins(void)
{
    int assigned = 0;
    ESP_LOGI(TAG, "---- Zang board pins (menuconfig -> Zang board pins) ----");
    for (size_t i = 0; i < BOARD_PIN_COUNT; i++) {
        const board_pin_t *p = &board_pin_table[i];
        if (p->gpio >= 0) {
            assigned++;
            ESP_LOGI(TAG, "  %-8s %-14s GPIO%d", p->group, p->name, p->gpio);
        } else {
            ESP_LOGI(TAG, "  %-8s %-14s --", p->group, p->name);
        }
    }
    ESP_LOGI(TAG, "---- %d of %d pins assigned ----", assigned, BOARD_PIN_COUNT);
}

esp_err_t board_init(void)
{
    board_log_pins();
    bool conflict = false;
    for (size_t i = 0; i < BOARD_PIN_COUNT; i++) {
        const board_pin_t *a = &board_pin_table[i];
        if (a->gpio < 0) continue;
        if (a->gpio > P4_GPIO_MAX) {
            ESP_LOGE(TAG, "%s = GPIO%d is out of range (0..%d)", a->name, a->gpio, P4_GPIO_MAX);
            conflict = true;
        }
        const char *why = NULL;
        if (pin_is_risky(a->gpio, &why)) ESP_LOGW(TAG, "%s = GPIO%d is a %s", a->name, a->gpio, why);
        for (size_t j = i + 1; j < BOARD_PIN_COUNT; j++) {
            if (board_pin_table[j].gpio == a->gpio) {
                ESP_LOGE(TAG, "GPIO%d assigned twice: %s and %s", a->gpio, a->name, board_pin_table[j].name);
                conflict = true;
            }
        }
    }
    if (conflict) return ESP_ERR_INVALID_STATE;

    if (BOARD_PIN_SERVO_PWR_EN >= 0) {
        /* Servo power starts OFF; motion turns it on after calibration is loaded. */
        gpio_config_t io = {.pin_bit_mask = BOARD_BIT(BOARD_PIN_SERVO_PWR_EN), .mode = GPIO_MODE_OUTPUT};
        ESP_RETURN_ON_ERROR(gpio_config(&io), TAG, "servo pwr gpio");
        s_servo_pwr_ready = true;
        board_servo_power(false);
    }
    if (BOARD_PIN_BTN_ESTOP >= 0) {
        gpio_config_t io = {.pin_bit_mask = BOARD_BIT(BOARD_PIN_BTN_ESTOP), .mode = GPIO_MODE_INPUT,
                            .pull_up_en = GPIO_PULLUP_ENABLE};
        ESP_RETURN_ON_ERROR(gpio_config(&io), TAG, "estop gpio");
    }
    return ESP_OK;
}

bool board_pins_assigned(const int *pins, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (pins[i] < 0) return false;
    }
    return true;
}

esp_err_t board_i2c_bus(i2c_master_bus_handle_t *out)
{
    ESP_RETURN_ON_FALSE(out, ESP_ERR_INVALID_ARG, TAG, "null out");
    if (BOARD_PIN_I2C_SDA < 0 || BOARD_PIN_I2C_SCL < 0) return ESP_ERR_NOT_SUPPORTED;
    if (!s_i2c_lock) {
        /* First caller wins; board_i2c_bus is called from init code on one task, the lock guards late callers. */
        s_i2c_lock = xSemaphoreCreateMutex();
        ESP_RETURN_ON_FALSE(s_i2c_lock, ESP_ERR_NO_MEM, TAG, "lock");
    }
    xSemaphoreTake(s_i2c_lock, portMAX_DELAY);
    esp_err_t err = ESP_OK;
    if (!s_i2c_bus) {
        i2c_master_bus_config_t cfg = {
            .i2c_port = CONFIG_ZANG_I2C_PORT,
            .sda_io_num = BOARD_PIN_I2C_SDA,
            .scl_io_num = BOARD_PIN_I2C_SCL,
            .clk_source = I2C_CLK_SRC_DEFAULT,
            .glitch_ignore_cnt = 7,
            .flags.enable_internal_pullup = true,
        };
        err = i2c_new_master_bus(&cfg, &s_i2c_bus);
    }
    xSemaphoreGive(s_i2c_lock);
    *out = s_i2c_bus;
    return err;
}

esp_err_t board_servo_power(bool on)
{
    if (!s_servo_pwr_ready) return ESP_OK;
#if CONFIG_ZANG_SERVO_PWR_ACTIVE_HIGH
    int level = on ? 1 : 0;
#else
    int level = on ? 0 : 1;
#endif
    return gpio_set_level(BOARD_PIN_SERVO_PWR_EN, level);
}

bool board_estop_pressed(void)
{
    if (BOARD_PIN_BTN_ESTOP < 0) return false;
    return gpio_get_level(BOARD_PIN_BTN_ESTOP) == 0;
}

/* ------------------------------------------------------------------ battery */
static adc_oneshot_unit_handle_t s_adc;
static adc_cali_handle_t s_adc_cali;
static adc_channel_t s_adc_chan;
static bool s_adc_tried;

static bool vbat_adc_init(void)
{
    s_adc_tried = true;
    adc_unit_t unit;
    if (adc_oneshot_io_to_channel(BOARD_PIN_VBAT_ADC, &unit, &s_adc_chan) != ESP_OK) {
        ESP_LOGW(TAG, "VBAT_ADC GPIO%d is not an ADC pin", BOARD_PIN_VBAT_ADC);
        return false;
    }
    adc_oneshot_unit_init_cfg_t ucfg = {.unit_id = unit};
    if (adc_oneshot_new_unit(&ucfg, &s_adc) != ESP_OK) return false;
    adc_oneshot_chan_cfg_t ccfg = {.atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_DEFAULT};
    if (adc_oneshot_config_channel(s_adc, s_adc_chan, &ccfg) != ESP_OK) return false;
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    adc_cali_curve_fitting_config_t cc = {.unit_id = unit, .chan = s_adc_chan, .atten = ADC_ATTEN_DB_12,
                                          .bitwidth = ADC_BITWIDTH_DEFAULT};
    if (adc_cali_create_scheme_curve_fitting(&cc, &s_adc_cali) != ESP_OK) s_adc_cali = NULL;
#endif
    return true;
}

uint16_t board_vbat_mv(void)
{
    if (BOARD_PIN_VBAT_ADC < 0) return 0;
    if (!s_adc && (s_adc_tried || !vbat_adc_init())) return 0;
    int raw = 0, mv = 0;
    if (adc_oneshot_read(s_adc, s_adc_chan, &raw) != ESP_OK) return 0;
    if (!s_adc_cali || adc_cali_raw_to_voltage(s_adc_cali, raw, &mv) != ESP_OK) mv = raw * 3300 / 4095;
    long v = (long)mv * CONFIG_ZANG_VBAT_DIVIDER_X1000 / 1000;
    return (uint16_t)(v > 65535 ? 65535 : v);
}
