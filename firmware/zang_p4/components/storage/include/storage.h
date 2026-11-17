/* storage.h - SD card mounted as FAT at STORAGE_ROOT ("/sdcard"). Layout: /sdcard/actions (.zact), /sdcard/audio (.wav). */
#pragma once

#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define STORAGE_ROOT "/sdcard"

/* ESP_ERR_NOT_SUPPORTED if SD pins are unassigned; ESP_ERR_NOT_FOUND if no card. */
esp_err_t storage_mount(void);
esp_err_t storage_unmount(void);
bool storage_is_mounted(void);

#ifdef __cplusplus
}
#endif
