/* zlink_server.c - UDP + TCP transports for ZLink (lwIP BSD sockets). */
#include "zlink_server.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "lwip/sockets.h"
#include "zang_tasks.h"

static const char *TAG = "zlink_srv";

struct zlink_server {
    zlink_server_config_t cfg;
    int udp_sock;
    int tcp_client; /* -1 = none */
    struct sockaddr_storage udp_peer;
    bool udp_peer_valid;
    uint16_t tx_seq;
    SemaphoreHandle_t tx_lock;
    zlink_server_stats_t stats;
    uint8_t udp_rx[ZLINK_MAX_FRAME];
    uint8_t tx_buf[ZLINK_MAX_FRAME];
    zlink_stream_t tcp_stream; /* large; lives in the heap-allocated server object */
};

static void udp_task(void *arg)
{
    struct zlink_server *s = arg;
    for (;;) {
        struct sockaddr_storage from;
        socklen_t flen = sizeof(from);
        int n = recvfrom(s->udp_sock, s->udp_rx, sizeof(s->udp_rx), 0, (struct sockaddr *)&from, &flen);
        if (n <= 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        zlink_frame_t f;
        size_t used;
        if (zlink_decode(s->udp_rx, (size_t)n, &f, &used) != ZLINK_OK || used != (size_t)n) {
            s->stats.rx_bad++;
            continue;
        }
        xSemaphoreTake(s->tx_lock, portMAX_DELAY);
        s->udp_peer = from;
        s->udp_peer_valid = true;
        xSemaphoreGive(s->tx_lock);
        s->stats.rx_frames++;
        if (s->cfg.on_frame) s->cfg.on_frame(&f, ZLINK_VIA_UDP, s->cfg.ctx);
    }
}

static void tcp_task(void *arg)
{
    struct zlink_server *s = arg;
    int listener = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(s->cfg.tcp_port),
                               .sin_addr.s_addr = htonl(INADDR_ANY)};
    int one = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (listener < 0 || bind(listener, (struct sockaddr *)&addr, sizeof(addr)) != 0 || listen(listener, 1) != 0) {
        ESP_LOGE(TAG, "TCP %u listen failed: errno %d", s->cfg.tcp_port, errno);
        vTaskDelete(NULL);
        return;
    }
    uint8_t chunk[512];
    for (;;) {
        int c = accept(listener, NULL, NULL);
        if (c < 0) continue;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        xSemaphoreTake(s->tx_lock, portMAX_DELAY);
        s->tcp_client = c;
        xSemaphoreGive(s->tx_lock);
        s->stats.tcp_clients++;
        zlink_stream_init(&s->tcp_stream);
        ESP_LOGI(TAG, "TCP client connected");
        for (;;) {
            int n = recv(c, chunk, sizeof(chunk), 0);
            if (n <= 0) break;
            size_t off = 0;
            while (off < (size_t)n) {
                off += zlink_stream_push(&s->tcp_stream, &chunk[off], (size_t)n - off);
                zlink_frame_t f;
                while (zlink_stream_next(&s->tcp_stream, &f)) {
                    s->stats.rx_frames++;
                    if (s->cfg.on_frame) s->cfg.on_frame(&f, ZLINK_VIA_TCP, s->cfg.ctx);
                }
            }
        }
        xSemaphoreTake(s->tx_lock, portMAX_DELAY);
        s->tcp_client = -1;
        xSemaphoreGive(s->tx_lock);
        close(c);
        ESP_LOGI(TAG, "TCP client gone (crc errors %lu)", (unsigned long)s->tcp_stream.crc_errors);
    }
}

esp_err_t zlink_server_start(const zlink_server_config_t *cfg, zlink_server_handle_t *out)
{
    ESP_RETURN_ON_FALSE(cfg && out, ESP_ERR_INVALID_ARG, TAG, "null arg");
    struct zlink_server *s = calloc(1, sizeof(*s));
    ESP_RETURN_ON_FALSE(s, ESP_ERR_NO_MEM, TAG, "no mem");
    s->cfg = *cfg;
    s->tcp_client = -1;
    s->tx_lock = xSemaphoreCreateMutex();
    s->udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(cfg->udp_port),
                               .sin_addr.s_addr = htonl(INADDR_ANY)};
    if (!s->tx_lock || s->udp_sock < 0 || bind(s->udp_sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        ESP_LOGE(TAG, "UDP %u bind failed: errno %d", cfg->udp_port, errno);
        if (s->udp_sock >= 0) close(s->udp_sock);
        if (s->tx_lock) vSemaphoreDelete(s->tx_lock);
        free(s);
        return ESP_FAIL;
    }
    if (zang_task_create(ZANG_TASK_ZLINK_UDP, udp_task, s, NULL) != pdPASS ||
        zang_task_create(ZANG_TASK_ZLINK_TCP, tcp_task, s, NULL) != pdPASS) {
        ESP_LOGE(TAG, "task create failed");
        return ESP_ERR_NO_MEM; /* tasks that did start keep running; this is a boot-time fatal path */
    }
    ESP_LOGI(TAG, "listening: UDP %u, TCP %u", cfg->udp_port, cfg->tcp_port);
    *out = s;
    return ESP_OK;
}

esp_err_t zlink_server_send(zlink_server_handle_t s, zlink_via_t via, uint8_t type, const uint8_t *payload,
                            uint16_t len)
{
    ESP_RETURN_ON_FALSE(s, ESP_ERR_INVALID_ARG, TAG, "null server");
    esp_err_t err = ESP_OK;
    xSemaphoreTake(s->tx_lock, portMAX_DELAY);
    size_t n = 0;
    if (zlink_encode(type, s->tx_seq++, payload, len, s->tx_buf, sizeof(s->tx_buf), &n) != ZLINK_OK) {
        err = ESP_ERR_INVALID_SIZE;
    } else if (via == ZLINK_VIA_UDP) {
        if (!s->udp_peer_valid) {
            err = ESP_ERR_INVALID_STATE;
        } else if (sendto(s->udp_sock, s->tx_buf, n, 0, (struct sockaddr *)&s->udp_peer,
                          sizeof(struct sockaddr_in)) != (int)n) {
            err = ESP_FAIL;
        }
    } else {
        if (s->tcp_client < 0) {
            err = ESP_ERR_INVALID_STATE;
        } else if (send(s->tcp_client, s->tx_buf, n, 0) != (int)n) {
            err = ESP_FAIL;
        }
    }
    if (err == ESP_OK) s->stats.tx_frames++;
    else if (err == ESP_FAIL) s->stats.tx_fail++;
    xSemaphoreGive(s->tx_lock);
    return err;
}

bool zlink_server_has_peer(zlink_server_handle_t s) { return s && s->udp_peer_valid; }

void zlink_server_get_stats(zlink_server_handle_t s, zlink_server_stats_t *out)
{
    if (s && out) *out = s->stats;
}
