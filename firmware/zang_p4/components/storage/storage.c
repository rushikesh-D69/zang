/* storage.c - SD card (SDMMC or SPI) + FAT. */
#include "storage.h"

#include "board_pins.h"
#include "board.h"
#include "driver/sdmmc_host.h"
#include "driver/sdspi_host.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "sdkconfig.h"
#include "sdmmc_cmd.h"
#if CONFIG_ZANG_SD_LDO_CHAN >= 0
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#endif

static const char *TAG = "storage";
static sdmmc_card_t *s_card;

esp_err_t storage_mount(void)
{
    if (s_card) return ESP_OK;
#if CONFIG_ZANG_SD_SPI
    if (!BOARD_PINS_ASSIGNED(BOARD_PIN_SD_CLK, BOARD_PIN_SD_CMD, BOARD_PIN_SD_D0, BOARD_PIN_SD_D3)) {
#elif CONFIG_ZANG_SD_SDMMC_4BIT
    if (!BOARD_PINS_ASSIGNED(BOARD_PIN_SD_CLK, BOARD_PIN_SD_CMD, BOARD_PIN_SD_D0, BOARD_PIN_SD_D1,
                             BOARD_PIN_SD_D2, BOARD_PIN_SD_D3)) {
#else
    if (!BOARD_PINS_ASSIGNED(BOARD_PIN_SD_CLK, BOARD_PIN_SD_CMD, BOARD_PIN_SD_D0)) {
#endif
        ESP_LOGW(TAG, "SD pins unassigned: storage disabled");
        return ESP_ERR_NOT_SUPPORTED;
    }

    const esp_vfs_fat_sdmmc_mount_config_t mount = {
        .format_if_mount_failed = false,
        .max_files = CONFIG_ZANG_SD_MAX_FILES,
        .allocation_unit_size = 16 * 1024,
    };
    esp_err_t err;
#if CONFIG_ZANG_SD_SPI
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = CONFIG_ZANG_SD_SPI_HOST;
    host.max_freq_khz = CONFIG_ZANG_SD_FREQ_KHZ;
    spi_bus_config_t bus = {.mosi_io_num = BOARD_PIN_SD_CMD, .miso_io_num = BOARD_PIN_SD_D0,
                            .sclk_io_num = BOARD_PIN_SD_CLK, .quadwp_io_num = -1, .quadhd_io_num = -1,
                            .max_transfer_sz = 4096};
    ESP_RETURN_ON_ERROR(spi_bus_initialize(host.slot, &bus, SPI_DMA_CH_AUTO), TAG, "spi bus");
    sdspi_device_config_t slot = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot.gpio_cs = BOARD_PIN_SD_D3;
    slot.gpio_cd = BOARD_PIN_SD_DET >= 0 ? BOARD_PIN_SD_DET : SDSPI_SLOT_NO_CD;
    slot.host_id = host.slot;
    err = esp_vfs_fat_sdspi_mount(STORAGE_ROOT, &host, &slot, &mount, &s_card);
#else
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.slot = CONFIG_ZANG_SD_SDMMC_SLOT;
    host.max_freq_khz = CONFIG_ZANG_SD_FREQ_KHZ;
#if CONFIG_ZANG_SD_LDO_CHAN >= 0
    sd_pwr_ctrl_ldo_config_t ldo = {.ldo_chan_id = CONFIG_ZANG_SD_LDO_CHAN};
    sd_pwr_ctrl_handle_t pwr = NULL;
    ESP_RETURN_ON_ERROR(sd_pwr_ctrl_new_on_chip_ldo(&ldo, &pwr), TAG, "sd ldo");
    host.pwr_ctrl_handle = pwr;
#endif
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
#if CONFIG_ZANG_SD_SDMMC_4BIT
    slot.width = 4;
#else
    slot.width = 1;
#endif
    /* Slot 1 routes through the GPIO matrix; slot 0 uses its fixed IOMUX pins and ignores these. */
    slot.clk = BOARD_PIN_SD_CLK;
    slot.cmd = BOARD_PIN_SD_CMD;
    slot.d0 = BOARD_PIN_SD_D0;
    slot.d1 = BOARD_PIN_SD_D1;
    slot.d2 = BOARD_PIN_SD_D2;
    slot.d3 = BOARD_PIN_SD_D3;
    slot.cd = BOARD_PIN_SD_DET >= 0 ? BOARD_PIN_SD_DET : SDMMC_SLOT_NO_CD;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
    err = esp_vfs_fat_sdmmc_mount(STORAGE_ROOT, &host, &slot, &mount, &s_card);
#endif
    if (err != ESP_OK) {
        s_card = NULL;
        ESP_LOGW(TAG, "mount failed: %s", esp_err_to_name(err));
        return err == ESP_FAIL ? ESP_ERR_NOT_FOUND : err;
    }
    ESP_LOGI(TAG, "SD mounted at %s: %s, %llu MB", STORAGE_ROOT, s_card->cid.name,
             ((unsigned long long)s_card->csd.capacity * s_card->csd.sector_size) >> 20);
    return ESP_OK;
}

esp_err_t storage_unmount(void)
{
    if (!s_card) return ESP_OK;
    esp_err_t err = esp_vfs_fat_sdcard_unmount(STORAGE_ROOT, s_card);
    s_card = NULL;
    return err;
}

bool storage_is_mounted(void) { return s_card != NULL; }
