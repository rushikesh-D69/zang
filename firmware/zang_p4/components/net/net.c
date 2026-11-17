/* net.c - Wi-Fi STA over ESP-Hosted (ESP32-C2 co-processor), mDNS advertisement, C2 reset helper. */
#include "net.h"

#include <string.h>
#include "board.h"
#include "driver/gpio.h"
#include "esp_check.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "mdns.h"
#include "nvs.h"
#include "sdkconfig.h"
#include "zlink.h"

/* The ESP-Hosted component owns its UART/reset options; they must match the board pin table. */
#if defined(CONFIG_ESP_HOSTED_UART_HOST_INTERFACE)
#if CONFIG_ESP_HOSTED_UART_PIN_TX != CONFIG_ZANG_PIN_C2_UART_TX || \
    CONFIG_ESP_HOSTED_UART_PIN_RX != CONFIG_ZANG_PIN_C2_UART_RX
#error "ESP-Hosted UART pins differ from CONFIG_ZANG_PIN_C2_UART_TX/RX (see docs/PINOUT.md)"
#endif
#if CONFIG_ESP_HOSTED_UART_GPIO_RESET_SLAVE != CONFIG_ZANG_PIN_C2_EN
#error "ESP-Hosted reset GPIO differs from CONFIG_ZANG_PIN_C2_EN"
#endif
#endif

static const char *TAG = "net";
#define BIT_GOT_IP BIT0
#define NVS_NS     "zang_net"

static EventGroupHandle_t s_ev;
static char s_ip[16];
static int s_retry;

static void on_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_ev, BIT_GOT_IP);
        s_ip[0] = '\0';
        int delay_ms = s_retry < 5 ? 500 : 5000; /* keep trying forever, slower after 5 attempts */
        s_retry++;
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = data;
        snprintf(s_ip, sizeof(s_ip), IPSTR, IP2STR(&e->ip_info.ip));
        s_retry = 0;
        ESP_LOGI(TAG, "connected, IP %s (%s.local)", s_ip, CONFIG_ZANG_HOSTNAME);
        xEventGroupSetBits(s_ev, BIT_GOT_IP);
    }
}

static bool load_credentials(char *ssid, size_t ssid_cap, char *pass, size_t pass_cap)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NS, NVS_READONLY, &h) == ESP_OK) {
        size_t a = ssid_cap, b = pass_cap;
        bool ok = nvs_get_str(h, "ssid", ssid, &a) == ESP_OK && nvs_get_str(h, "pass", pass, &b) == ESP_OK;
        nvs_close(h);
        if (ok && ssid[0]) return true;
    }
    strlcpy(ssid, CONFIG_ZANG_WIFI_SSID, ssid_cap);
    strlcpy(pass, CONFIG_ZANG_WIFI_PASSWORD, pass_cap);
    return ssid[0] != '\0';
}

esp_err_t net_set_credentials(const char *ssid, const char *password)
{
    ESP_RETURN_ON_FALSE(ssid && password && strlen(ssid) < 32 && strlen(password) < 64, ESP_ERR_INVALID_ARG, TAG,
                        "bad credentials");
    nvs_handle_t h;
    ESP_RETURN_ON_ERROR(nvs_open(NVS_NS, NVS_READWRITE, &h), TAG, "nvs");
    esp_err_t err = nvs_set_str(h, "ssid", ssid);
    if (err == ESP_OK) err = nvs_set_str(h, "pass", password);
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err;
}

static void start_mdns(void)
{
    if (mdns_init() != ESP_OK) return;
    mdns_hostname_set(CONFIG_ZANG_HOSTNAME);
    mdns_instance_name_set("Zang humanoid");
    mdns_service_add(NULL, "_zlink", "_udp", ZLINK_UDP_PORT, NULL, 0);
    mdns_service_add(NULL, "_zlink", "_tcp", ZLINK_TCP_PORT, NULL, 0);
}

esp_err_t net_start(void)
{
    char ssid[33], pass[65];
    if (!load_credentials(ssid, sizeof(ssid), pass, sizeof(pass))) {
        ESP_LOGW(TAG, "no Wi-Fi SSID (NVS or CONFIG_ZANG_WIFI_SSID): network disabled");
        return ESP_ERR_NOT_FOUND;
    }
    s_ev = xEventGroupCreate();
    ESP_RETURN_ON_FALSE(s_ev, ESP_ERR_NO_MEM, TAG, "event group");
    if (BOARD_PIN_C2_BOOT >= 0) { /* make sure the C2 boots its firmware, not ROM download mode */
        gpio_config_t io = {.pin_bit_mask = BOARD_BIT(BOARD_PIN_C2_BOOT), .mode = GPIO_MODE_OUTPUT};
        gpio_config(&io);
        gpio_set_level(BOARD_PIN_C2_BOOT, 1);
    }
    ESP_RETURN_ON_ERROR(esp_netif_init(), TAG, "netif");
    esp_err_t err = esp_event_loop_create_default();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) return err;
    esp_netif_t *sta = esp_netif_create_default_wifi_sta();
    esp_netif_set_hostname(sta, CONFIG_ZANG_HOSTNAME);
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_RETURN_ON_ERROR(esp_wifi_init(&cfg), TAG, "wifi init (is the C2 running zang_c2_hosted?)");
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL));
    wifi_config_t wc = {0};
    strlcpy((char *)wc.sta.ssid, ssid, sizeof(wc.sta.ssid));
    strlcpy((char *)wc.sta.password, pass, sizeof(wc.sta.password));
    wc.sta.threshold.authmode = pass[0] ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
    ESP_RETURN_ON_ERROR(esp_wifi_set_mode(WIFI_MODE_STA), TAG, "mode");
    ESP_RETURN_ON_ERROR(esp_wifi_set_config(WIFI_IF_STA, &wc), TAG, "config");
    ESP_RETURN_ON_ERROR(esp_wifi_start(), TAG, "start");
    esp_wifi_set_ps(WIFI_PS_NONE); /* latency over power: ZLink is real-time */
    start_mdns();
    ESP_LOGI(TAG, "connecting to \"%s\" via ESP32-C2 (ESP-Hosted UART %d baud)", ssid,
             CONFIG_ESP_HOSTED_UART_BAUDRATE);
    return ESP_OK;
}

esp_err_t net_wait_connected(uint32_t timeout_ms)
{
    if (!s_ev) return ESP_ERR_INVALID_STATE;
    EventBits_t b = xEventGroupWaitBits(s_ev, BIT_GOT_IP, pdFALSE, pdTRUE, pdMS_TO_TICKS(timeout_ms));
    return (b & BIT_GOT_IP) ? ESP_OK : ESP_ERR_TIMEOUT;
}

bool net_is_connected(void) { return s_ev && (xEventGroupGetBits(s_ev) & BIT_GOT_IP); }

const char *net_ip_str(void) { return s_ip; }

esp_err_t net_c2_reset(bool download_mode)
{
    ESP_RETURN_ON_FALSE(BOARD_PIN_C2_EN >= 0, ESP_ERR_NOT_SUPPORTED, TAG, "C2_EN unassigned");
    gpio_config_t io = {.pin_bit_mask = BOARD_BIT(BOARD_PIN_C2_EN), .mode = GPIO_MODE_OUTPUT};
    if (BOARD_PIN_C2_BOOT >= 0) io.pin_bit_mask |= BOARD_BIT(BOARD_PIN_C2_BOOT);
    ESP_RETURN_ON_ERROR(gpio_config(&io), TAG, "gpio");
    if (BOARD_PIN_C2_BOOT >= 0) gpio_set_level(BOARD_PIN_C2_BOOT, download_mode ? 0 : 1);
    gpio_set_level(BOARD_PIN_C2_EN, 0);
    vTaskDelay(pdMS_TO_TICKS(20));
    gpio_set_level(BOARD_PIN_C2_EN, 1); /* strap is latched on this edge */
    vTaskDelay(pdMS_TO_TICKS(50));
    if (BOARD_PIN_C2_BOOT >= 0 && download_mode) {
        ESP_LOGI(TAG, "C2 is in ROM download mode (UART0 GPIO19/20 <-> P4 GPIO%d/%d)", BOARD_PIN_C2_UART_TX,
                 BOARD_PIN_C2_UART_RX);
    }
    return ESP_OK;
}
