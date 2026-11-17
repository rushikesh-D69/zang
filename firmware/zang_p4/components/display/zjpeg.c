/* zjpeg.c - hardware JPEG decode (RGB565) + scale to the panel. */
#include "zjpeg.h"

#include "driver/jpeg_decode.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static const char *TAG = "zjpeg";

#define MAX_W 1280
#define MAX_H 720

static struct {
    SemaphoreHandle_t lock;
    jpeg_decoder_handle_t dec;
    uint16_t *out;
    size_t out_cap;
} z;

void *zjpeg_alloc_input(size_t n)
{
    jpeg_decode_memory_alloc_cfg_t cfg = {.buffer_direction = JPEG_DEC_ALLOC_INPUT_BUFFER};
    size_t got = 0;
    return jpeg_alloc_decoder_mem(n, &cfg, &got);
}

void zjpeg_free(void *p) { heap_caps_free(p); }

static esp_err_t ensure_engine(void)
{
    if (!z.lock) z.lock = xSemaphoreCreateMutex();
    ESP_RETURN_ON_FALSE(z.lock, ESP_ERR_NO_MEM, TAG, "lock");
    if (z.dec) return ESP_OK;
    const jpeg_decode_engine_cfg_t cfg = {.timeout_ms = 200};
    return jpeg_new_decoder_engine(&cfg, &z.dec);
}

esp_err_t zjpeg_decode_to_panel(const uint8_t *jpg, size_t len, img_fit_t mode, uint16_t bg, uint16_t *dst, int dw,
                                int dh)
{
    ESP_RETURN_ON_FALSE(jpg && len > 4 && dst, ESP_ERR_INVALID_ARG, TAG, "bad arg");
    jpeg_decode_picture_info_t info;
    ESP_RETURN_ON_ERROR(jpeg_decoder_get_info(jpg, (uint32_t)len, &info), TAG, "bad JPEG header");
    ESP_RETURN_ON_FALSE(info.width >= 8 && info.height >= 8 && info.width <= MAX_W && info.height <= MAX_H,
                        ESP_ERR_INVALID_ARG, TAG, "unsupported size %lux%lu", (unsigned long)info.width,
                        (unsigned long)info.height);
    /* the hardware writes whole 16x16 blocks */
    const size_t stride = (info.width + 15u) & ~15u, rows = (info.height + 15u) & ~15u;
    const size_t need = stride * rows * 2;

    if (!z.lock) z.lock = xSemaphoreCreateMutex();
    ESP_RETURN_ON_FALSE(z.lock, ESP_ERR_NO_MEM, TAG, "lock");
    xSemaphoreTake(z.lock, portMAX_DELAY);
    esp_err_t err = ensure_engine();
    if (err == ESP_OK && z.out_cap < need) {
        heap_caps_free(z.out);
        jpeg_decode_memory_alloc_cfg_t mc = {.buffer_direction = JPEG_DEC_ALLOC_OUTPUT_BUFFER};
        size_t got = 0;
        z.out = jpeg_alloc_decoder_mem(need, &mc, &got);
        z.out_cap = z.out ? got : 0;
        if (!z.out) err = ESP_ERR_NO_MEM;
    }
    if (err == ESP_OK) {
        const jpeg_decode_cfg_t dc = {.output_format = JPEG_DECODE_OUT_FORMAT_RGB565,
                                      .rgb_order = JPEG_DEC_RGB_ELEMENT_ORDER_RGB,
                                      .conv_std = JPEG_YUV_RGB_CONV_STD_BT601};
        uint32_t out_len = 0;
        err = jpeg_decoder_process(z.dec, &dc, jpg, (uint32_t)len, (uint8_t *)z.out, (uint32_t)z.out_cap, &out_len);
        if (err == ESP_OK) {
            img_scale_to_panel(z.out, (int)info.width, (int)info.height, (int)stride, dst, dw, dh, mode, bg);
        } else {
            ESP_LOGW(TAG, "decode failed: %s (%lux%lu, %u bytes)", esp_err_to_name(err), (unsigned long)info.width,
                     (unsigned long)info.height, (unsigned)len);
        }
    }
    xSemaphoreGive(z.lock);
    return err;
}
