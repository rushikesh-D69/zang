/*
 * Zang P4 firmware entry point. Only wiring here: every driver lives in components/ and is optional.
 * A driver whose pins are unassigned (or whose hardware is absent) is skipped; the robot still boots.
 */
#include "app.h"
#include "board.h"
#include "esp_log.h"
#include "net.h"
#include "nvs_flash.h"
#include "servo_bus.h"
#include "storage.h"
#include "zlink.h"

static const char *TAG = "main";

static void init_nvs(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS layout changed: erasing (calibration must be redone)");
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
}

#define TRY(what, call) \
    do { \
        esp_err_t _e = (call); \
        if (_e != ESP_OK) ESP_LOGW(TAG, "%s: %s", what, esp_err_to_name(_e)); \
    } while (0)

void app_main(void)
{
    ESP_LOGI(TAG, "Zang P4 firmware (" ROBOT_NAME ", %d joints, board v0.0.1)", ROBOT_N_JOINTS);
    init_nvs();
    app_ctx_t ctx = {0};
    ctx.board_ok = board_init() == ESP_OK;
    if (!ctx.board_ok) ESP_LOGE(TAG, "board pin conflict: drivers skipped, robot stays in ESTOP");

    if (ctx.board_ok) {
        TRY("storage", storage_mount());
        TRY("display", display_init(&ctx.display));
#if CONFIG_ZANG_LCD_SELF_TEST
        if (ctx.display) display_self_test(ctx.display, NULL);
#endif
        TRY("audio", audio_init(&ctx.audio));
        TRY("camera", camera_init(&ctx.camera));
        TRY("imu", imu_init(&ctx.imu));
    }

    servo_bus_handle_t bus = NULL;
    if (ctx.board_ok) {
        servo_bus_config_t bcfg = servo_bus_default_config();
        TRY("servo bus", servo_bus_create(&bcfg, &bus));
    }
    app_set_audio(ctx.audio);
    motion_config_t mcfg = {.bus = bus};
    app_motion_config(&mcfg);
    TRY("motion", motion_start(&mcfg, &ctx.motion));

    TRY("wifi", net_start());
    TRY("app", app_start(&ctx));
    if (net_wait_connected(15000) == ESP_OK) {
        ESP_LOGI(TAG, "ready: ZLink on %s UDP %d / TCP %d", net_ip_str(), ZLINK_UDP_PORT, ZLINK_TCP_PORT);
    }
}
