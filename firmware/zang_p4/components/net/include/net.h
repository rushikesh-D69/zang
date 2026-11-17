/*
 * net.h - Wi-Fi station through the ESP32-C2 co-processor (ESP-Hosted over UART + esp_wifi_remote), mDNS.
 * Credentials: NVS ("zang_net") first, Kconfig defaults second. mDNS: <hostname>.local, _zlink._udp:4210.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Start Wi-Fi (non-blocking). ESP_ERR_NOT_FOUND if no SSID is configured anywhere. */
esp_err_t net_start(void);
/* Wait for an IP. ESP_ERR_TIMEOUT on timeout. */
esp_err_t net_wait_connected(uint32_t timeout_ms);
bool net_is_connected(void);
/* Persist credentials in NVS (used from the next net_start / reboot). */
esp_err_t net_set_credentials(const char *ssid, const char *password);
/* Dotted IPv4 address, "" if not connected. */
const char *net_ip_str(void);

/*
 * Reset the C2 through EN_C2, optionally holding BOOT_C2 low to enter its ROM download mode (for flashing the
 * C2 from the P4 with esp-serial-flasher later). Do not call while ESP-Hosted is running unless you restart it.
 */
esp_err_t net_c2_reset(bool download_mode);

#ifdef __cplusplus
}
#endif
