/* camera.c - OV2640 (esp_cam_sensor) on the ESP32-P4 DVP controller, double-buffered frame delivery. */
#include "camera.h"

#include "sdkconfig.h"
#include "esp_check.h"
#include "esp_log.h"

static const char *TAG = "camera";

#if !CONFIG_ZANG_CAMERA_ENABLE

esp_err_t camera_init(camera_handle_t *out)
{
    ESP_LOGI(TAG, "camera disabled in Kconfig");
    return ESP_ERR_NOT_SUPPORTED;
}
esp_err_t camera_start(camera_handle_t cam, camera_frame_cb_t cb, void *ctx) { return ESP_ERR_NOT_SUPPORTED; }
esp_err_t camera_stop(camera_handle_t cam) { return ESP_ERR_NOT_SUPPORTED; }
esp_err_t camera_get_stats(camera_handle_t cam, camera_stats_t *out) { return ESP_ERR_NOT_SUPPORTED; }
esp_err_t camera_configure(camera_handle_t cam, camera_size_t size, uint8_t quality) { return ESP_ERR_NOT_SUPPORTED; }

#else

#include <stdlib.h>
#include <string.h>
#include "board.h"
#include "esp_attr.h"
#include "esp_cam_ctlr.h"
#include "esp_cam_ctlr_dvp.h"
#include "esp_cam_sensor.h"
#include "esp_cam_sensor_detect.h"
#include "esp_heap_caps.h"
#include "esp_sccb_i2c.h"
#include "esp_sccb_intf.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "zang_tasks.h"

/* The OV2640 compresses to JPEG itself: QVGA 320x240 (50 fps) or VGA 640x480 (25 fps). The DVP controller's frame
 * buffer for JPEG is h_res * v_res bytes, a safe upper bound for these sizes at any quality. */
static const struct {
    int w, h;
    const char *format;
} SIZES[CAMERA_SIZE_COUNT] = {
    [CAMERA_SIZE_QVGA] = {320, 240, "DVP_8bit_20Minput_JPEG_320x240_50fps"},
    [CAMERA_SIZE_VGA] = {640, 480, "DVP_8bit_20Minput_JPEG_640x480_25fps"},
};
#define N_BUF 2

struct camera {
    esp_cam_ctlr_handle_t ctlr;
    esp_cam_sensor_device_t *sensor;
    uint8_t *buf[N_BUF];
    size_t buf_len;
    volatile uint32_t busy; /* bit i: buffer i is being filled or is held by the consumer */
    portMUX_TYPE mux;
    QueueHandle_t done;     /* esp_cam_ctlr_trans_t of finished frames */
    SemaphoreHandle_t cfg_lock; /* held by the camera task while the callback uses a frame, and by camera_configure */
    camera_frame_cb_t cb;
    void *ctx;
    TaskHandle_t task;
    bool running, up;
    camera_size_t size;
    uint8_t quality; /* 1..100 */
    uint32_t frames, dropped;
    int64_t t_first;
};

static bool IRAM_ATTR on_get_new_trans(esp_cam_ctlr_handle_t h, esp_cam_ctlr_trans_t *t, void *arg)
{
    struct camera *c = arg;
    portENTER_CRITICAL_ISR(&c->mux);
    for (int i = 0; i < N_BUF; i++) {
        if (!(c->busy & (1u << i))) {
            c->busy |= 1u << i;
            t->buffer = c->buf[i];
            t->buflen = c->buf_len;
            break;
        }
    }
    if (!t->buffer) c->dropped++; /* driver falls back to its backup buffer: this frame is dropped */
    portEXIT_CRITICAL_ISR(&c->mux);
    return false;
}

static bool IRAM_ATTR on_trans_finished(esp_cam_ctlr_handle_t h, esp_cam_ctlr_trans_t *t, void *arg)
{
    struct camera *c = arg;
    BaseType_t woken = pdFALSE;
    if (xQueueSendFromISR(c->done, t, &woken) != pdTRUE) {
        portENTER_CRITICAL_ISR(&c->mux);
        for (int i = 0; i < N_BUF; i++) {
            if (t->buffer == c->buf[i]) c->busy &= ~(1u << i);
        }
        c->dropped++;
        portEXIT_CRITICAL_ISR(&c->mux);
    }
    return woken == pdTRUE;
}

static void camera_task(void *arg)
{
    struct camera *c = arg;
    esp_cam_ctlr_trans_t t;
    for (;;) {
        if (xQueueReceive(c->done, &t, portMAX_DELAY) != pdTRUE) continue;
        xSemaphoreTake(c->cfg_lock, portMAX_DELAY);
        int64_t now = esp_timer_get_time();
        if (!c->frames) c->t_first = now;
        camera_frame_t f = {.data = t.buffer, .len = t.received_size, .width = SIZES[c->size].w,
                            .height = SIZES[c->size].h, .jpeg = true, .t_us = now, .seq = c->frames++};
        if (c->cb && c->running && c->up && t.received_size > 4) c->cb(&f, c->ctx);
        portENTER_CRITICAL(&c->mux);
        for (int i = 0; i < N_BUF; i++) {
            if (t.buffer == c->buf[i]) c->busy &= ~(1u << i);
        }
        portEXIT_CRITICAL(&c->mux);
        xSemaphoreGive(c->cfg_lock);
    }
}

static esp_err_t detect_sensor(struct camera *c)
{
    i2c_master_bus_handle_t bus;
    ESP_RETURN_ON_ERROR(board_i2c_bus(&bus), TAG, "I2C bus (SCCB) unavailable");
    esp_cam_sensor_config_t cfg = {.reset_pin = BOARD_PIN_CAM_RESET, .pwdn_pin = BOARD_PIN_CAM_PWDN,
                                   .xclk_pin = -1 /* XCLK comes from the DVP controller */};
    for (esp_cam_sensor_detect_fn_t *p = &__esp_cam_sensor_detect_fn_array_start;
         p < &__esp_cam_sensor_detect_fn_array_end; ++p) {
        if (p->port != ESP_CAM_SENSOR_DVP) continue;
        sccb_i2c_config_t sc = {.scl_speed_hz = CONFIG_ZANG_CAMERA_SCCB_HZ, .device_address = p->sccb_addr,
                                .dev_addr_length = I2C_ADDR_BIT_LEN_7};
        ESP_RETURN_ON_ERROR(sccb_new_i2c_io(bus, &sc, &cfg.sccb_handle), TAG, "sccb");
        cfg.sensor_port = p->port;
        c->sensor = p->detect(&cfg);
        if (c->sensor) return ESP_OK;
        esp_sccb_del_i2c_io(cfg.sccb_handle);
    }
    return ESP_ERR_NOT_FOUND;
}

static esp_err_t set_quality(struct camera *c)
{
    int32_t q = 1 + ((int32_t)c->quality - 1) * 59 / 99; /* 1..100 -> the OV2640's 1..60, higher = better */
    return esp_cam_sensor_set_para_value(c->sensor, ESP_CAM_SENSOR_JPEG_QUALITY, &q, sizeof(q));
}

static esp_err_t configure_sensor(struct camera *c)
{
    int off = 0;
    esp_cam_sensor_ioctl(c->sensor, ESP_CAM_SENSOR_IOC_S_STREAM, &off);
    /* Soft reset over SCCB (OV2640 COM7 = 0x80): v0.0.1 has no RESET line and PWDN floats. */
    esp_cam_sensor_ioctl(c->sensor, ESP_CAM_SENSOR_IOC_SW_RESET, NULL);
    vTaskDelay(pdMS_TO_TICKS(10));
    esp_cam_sensor_format_array_t fmts = {0};
    ESP_RETURN_ON_ERROR(esp_cam_sensor_query_format(c->sensor, &fmts), TAG, "query formats");
    const esp_cam_sensor_format_t *want = NULL;
    for (uint32_t i = 0; i < fmts.count; i++) {
        if (strcmp(fmts.format_array[i].name, SIZES[c->size].format) == 0) want = &fmts.format_array[i];
    }
    if (!want) {
        ESP_LOGE(TAG, "sensor has no format %s (enable it in sdkconfig: CONFIG_CAMERA_OV2640_DVP_JPEG_*)",
                 SIZES[c->size].format);
        return ESP_ERR_NOT_SUPPORTED;
    }
    ESP_RETURN_ON_ERROR(esp_cam_sensor_set_format(c->sensor, want), TAG, "set format");
    esp_err_t qe = set_quality(c);
    if (qe != ESP_OK) ESP_LOGW(TAG, "JPEG quality not applied: %s", esp_err_to_name(qe));
    int on = 1;
    return esp_cam_sensor_ioctl(c->sensor, ESP_CAM_SENSOR_IOC_S_STREAM, &on);
}

/* Tear the controller down (frames in flight are lost). Caller holds cfg_lock. */
static void ctlr_down(struct camera *c)
{
    if (!c->up) return;
    esp_cam_ctlr_stop(c->ctlr);
    esp_cam_ctlr_disable(c->ctlr);
    esp_cam_ctlr_del(c->ctlr);
    c->ctlr = NULL;
    for (int i = 0; i < N_BUF; i++) {
        heap_caps_free(c->buf[i]);
        c->buf[i] = NULL;
    }
    esp_cam_ctlr_trans_t drop;
    while (xQueueReceive(c->done, &drop, 0) == pdTRUE) {}
    c->busy = 0;
    c->up = false;
}

/* DVP controller (drives XCLK) -> sensor -> buffers -> enable. Caller holds cfg_lock (or runs before the task). */
static esp_err_t ctlr_up(struct camera *c)
{
    const int w = SIZES[c->size].w, h = SIZES[c->size].h;
    /* 1. DVP controller first: it drives XCLK, without which the sensor does not answer on SCCB. */
    esp_cam_ctlr_dvp_pin_config_t pins = {
        .data_width = CAM_CTLR_DATA_WIDTH_8,
        .data_io = {BOARD_PIN_CAM_D0, BOARD_PIN_CAM_D1, BOARD_PIN_CAM_D2, BOARD_PIN_CAM_D3, BOARD_PIN_CAM_D4,
                    BOARD_PIN_CAM_D5, BOARD_PIN_CAM_D6, BOARD_PIN_CAM_D7},
        .vsync_io = BOARD_PIN_CAM_VSYNC,
        .de_io = BOARD_PIN_CAM_HREF,
        .pclk_io = BOARD_PIN_CAM_PCLK,
        .xclk_io = BOARD_PIN_CAM_XCLK,
    };
    esp_cam_ctlr_dvp_config_t dvp = {
        .ctlr_id = 0,
        .clk_src = CAM_CLK_SRC_DEFAULT,
        .h_res = w,
        .v_res = h,
        .input_data_color_type = CAM_CTLR_COLOR_RGB565, /* ignored for JPEG */
        .pic_format_jpeg = 1,
        .dma_burst_size = 64,
        .xclk_freq = CONFIG_ZANG_CAMERA_XCLK_HZ,
        .pin = &pins,
    };
    ESP_RETURN_ON_ERROR(esp_cam_new_dvp_ctlr(&dvp, &c->ctlr), TAG, "dvp controller");
    vTaskDelay(pdMS_TO_TICKS(20)); /* sensor PLL lock on XCLK */

    /* 2. Sensor over SCCB. */
    if (!c->sensor) {
        esp_err_t err = detect_sensor(c);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "no camera sensor answered on SCCB (check PWDN rework / module)");
            esp_cam_ctlr_del(c->ctlr);
            c->ctlr = NULL;
            return err;
        }
    }
    esp_err_t err = configure_sensor(c);
    if (err != ESP_OK) {
        esp_cam_ctlr_del(c->ctlr);
        c->ctlr = NULL;
        return err;
    }

    /* 3. Frame buffers + callbacks. */
    c->buf_len = (size_t)w * h;
    for (int i = 0; i < N_BUF; i++) {
        c->buf[i] = esp_cam_ctlr_alloc_buffer(c->ctlr, c->buf_len, MALLOC_CAP_SPIRAM);
        if (!c->buf[i]) {
            for (int k = 0; k < i; k++) heap_caps_free(c->buf[k]), c->buf[k] = NULL;
            esp_cam_ctlr_del(c->ctlr);
            c->ctlr = NULL;
            return ESP_ERR_NO_MEM;
        }
    }
    esp_cam_ctlr_evt_cbs_t cbs = {.on_get_new_trans = on_get_new_trans, .on_trans_finished = on_trans_finished};
    ESP_RETURN_ON_ERROR(esp_cam_ctlr_register_event_callbacks(c->ctlr, &cbs, c), TAG, "callbacks");
    ESP_RETURN_ON_ERROR(esp_cam_ctlr_enable(c->ctlr), TAG, "enable");
    c->up = true;
    return ESP_OK;
}

esp_err_t camera_init(camera_handle_t *out)
{
    ESP_RETURN_ON_FALSE(out, ESP_ERR_INVALID_ARG, TAG, "null");
    if (!BOARD_PINS_ASSIGNED(BOARD_PIN_CAM_D0, BOARD_PIN_CAM_D1, BOARD_PIN_CAM_D2, BOARD_PIN_CAM_D3,
                             BOARD_PIN_CAM_D4, BOARD_PIN_CAM_D5, BOARD_PIN_CAM_D6, BOARD_PIN_CAM_D7,
                             BOARD_PIN_CAM_PCLK, BOARD_PIN_CAM_XCLK, BOARD_PIN_CAM_HREF, BOARD_PIN_CAM_VSYNC)) {
        ESP_LOGW(TAG, "DVP pins unassigned: camera disabled");
        return ESP_ERR_NOT_SUPPORTED;
    }
    struct camera *c = heap_caps_calloc(1, sizeof(*c), MALLOC_CAP_INTERNAL); /* touched from ISRs */
    ESP_RETURN_ON_FALSE(c, ESP_ERR_NO_MEM, TAG, "no mem");
    portMUX_INITIALIZE(&c->mux);
    c->done = xQueueCreate(N_BUF, sizeof(esp_cam_ctlr_trans_t));
    c->cfg_lock = xSemaphoreCreateMutex();
    ESP_RETURN_ON_FALSE(c->done && c->cfg_lock, ESP_ERR_NO_MEM, TAG, "queue");
    c->size = CAMERA_SIZE_QVGA;
    c->quality = CONFIG_ZANG_CAMERA_DEFAULT_QUALITY;
    esp_err_t err = ctlr_up(c);
    if (err != ESP_OK) {
        vQueueDelete(c->done);
        vSemaphoreDelete(c->cfg_lock);
        free(c);
        return err;
    }
    if (zang_task_create(ZANG_TASK_CAMERA, camera_task, c, &c->task) != pdPASS) return ESP_ERR_NO_MEM;
    ESP_LOGI(TAG, "%s detected, JPEG %s, XCLK %d Hz", esp_cam_sensor_get_name(c->sensor), SIZES[c->size].format,
             CONFIG_ZANG_CAMERA_XCLK_HZ);
    *out = c;
    return ESP_OK;
}

esp_err_t camera_start(camera_handle_t c, camera_frame_cb_t cb, void *ctx)
{
    ESP_RETURN_ON_FALSE(c, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(c->cfg_lock, portMAX_DELAY);
    c->cb = cb;
    c->ctx = ctx;
    esp_err_t err = c->up ? ESP_OK : ctlr_up(c);
    if (err == ESP_OK && !c->running) err = esp_cam_ctlr_start(c->ctlr);
    c->running = err == ESP_OK;
    xSemaphoreGive(c->cfg_lock);
    return err;
}

esp_err_t camera_stop(camera_handle_t c)
{
    ESP_RETURN_ON_FALSE(c, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(c->cfg_lock, portMAX_DELAY);
    esp_err_t err = ESP_OK;
    if (c->running && c->up) err = esp_cam_ctlr_stop(c->ctlr);
    c->running = false;
    xSemaphoreGive(c->cfg_lock);
    return err;
}

esp_err_t camera_configure(camera_handle_t c, camera_size_t size, uint8_t quality)
{
    ESP_RETURN_ON_FALSE(c && size < CAMERA_SIZE_COUNT, ESP_ERR_INVALID_ARG, TAG, "bad arg");
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;
    xSemaphoreTake(c->cfg_lock, portMAX_DELAY);
    esp_err_t err = ESP_OK;
    bool was_running = c->running;
    if (size != c->size) { /* new resolution: rebuild the controller and re-select the sensor format */
        ctlr_down(c);
        c->size = size;
        c->quality = quality;
        err = ctlr_up(c);
        if (err == ESP_OK && was_running) err = esp_cam_ctlr_start(c->ctlr);
        c->running = was_running && err == ESP_OK;
    } else if (quality != c->quality) {
        c->quality = quality;
        err = c->up ? set_quality(c) : ESP_OK;
    }
    xSemaphoreGive(c->cfg_lock);
    return err;
}

esp_err_t camera_get_stats(camera_handle_t c, camera_stats_t *out)
{
    ESP_RETURN_ON_FALSE(c && out, ESP_ERR_INVALID_ARG, TAG, "null");
    int64_t dt = esp_timer_get_time() - c->t_first;
    out->frames = c->frames;
    out->dropped = c->dropped;
    out->fps = (c->frames > 1 && dt > 0) ? (float)(c->frames - 1) * 1e6f / (float)dt : 0.0f;
    out->sensor = esp_cam_sensor_get_name(c->sensor);
    out->format = SIZES[c->size].format;
    return ESP_OK;
}

#endif /* CONFIG_ZANG_CAMERA_ENABLE */
