/* display.c - SPI TFT bring-up, face/image/text/camera/status modes, dirty-rect flushing via two DMA line buffers. */
#include "display.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "board.h"
#include "canvas.h"
#include "driver/ledc.h"
#include "driver/spi_master.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_log.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "face_sdf.h"
#include "faces_lib.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "zang_tasks.h"
#include "zjpeg.h"
#if CONFIG_ZANG_LCD_ILI9341
#include "esp_lcd_ili9341.h"
#elif CONFIG_ZANG_LCD_GC9A01
#include "esp_lcd_gc9a01.h"
#endif

static const char *TAG = "display";

#define W        CONFIG_ZANG_LCD_WIDTH
#define H        CONFIG_ZANG_LCD_HEIGHT
#define LINES    CONFIG_ZANG_LCD_LINES_PER_BUF
#define LCD_HOST SPI2_HOST
#define BL_TIMER LEDC_TIMER_1
#define BL_CHAN  LEDC_CHANNEL_1
#define FACE_PX  240 /* the face / images are 240 x 240, centred on the panel */
#define OX       ((W - FACE_PX) / 2)
#define OY       ((H - FACE_PX) / 2)
#define TEXT_MAX 256
_Static_assert(W >= FACE_PX && H >= FACE_PX, "the face renderer needs a panel of at least 240 x 240");
#if CONFIG_ZANG_LCD_ILI9341
#define PANEL_NAME "ILI9341"
#elif CONFIG_ZANG_LCD_GC9A01
#define PANEL_NAME "GC9A01"
#else
#define PANEL_NAME "ST7789"
#endif

struct display {
    esp_lcd_panel_io_handle_t io;
    esp_lcd_panel_handle_t panel;
    canvas_t canvas;
    uint16_t *line[2];
    int next_line;
    SemaphoreHandle_t inflight; /* counting, max 2: one token per free DMA line buffer */
    SemaphoreHandle_t lock;     /* everything below */
    bool paused;

    display_mode_t mode;
    bool mode_dirty;            /* the next task pass redraws the whole screen for the current mode */
    /* face */
    face_engine_t eng;
    face_frame_t last_frame;
    face_rect_t last_bbox;
    display_stream_fn_t stream_fn;
    void *stream_ctx;
    uint8_t backlight;
    /* image */
    uint16_t *img;              /* 240x240 panel byte order */
    uint16_t img_id;
    bool img_valid;
    /* camera */
    uint16_t *cam;
    int64_t cam_last_us;
    /* text */
    char text[TEXT_MAX];
    int text_n;
    uint8_t text_size;
    uint16_t text_fg, text_bg;
    int64_t text_until_ms;      /* 0 = until changed */
    char status[64];
};

static inline uint32_t now_ms32(void) { return (uint32_t)(esp_timer_get_time() / 1000); }

static bool on_trans_done(esp_lcd_panel_io_handle_t io, esp_lcd_panel_io_event_data_t *e, void *ctx)
{
    struct display *d = ctx;
    BaseType_t woken = pdFALSE;
    xSemaphoreGiveFromISR(d->inflight, &woken);
    return woken == pdTRUE;
}

/* Send the canvas' dirty rectangle (caller holds d->lock). Transfers complete in order, so taking a token
 * guarantees the buffer used two transfers ago is free again. */
static void flush_locked(struct display *d)
{
    canvas_t *c = &d->canvas;
    if (!c->has_dirty) return;
    int x0 = c->dx0, x1 = c->dx1 + 1, w = x1 - x0;
    for (int y = c->dy0; y <= c->dy1; y += LINES) {
        int n = c->dy1 + 1 - y;
        if (n > LINES) n = LINES;
        xSemaphoreTake(d->inflight, portMAX_DELAY);
        uint16_t *buf = d->line[d->next_line];
        d->next_line ^= 1;
        for (int k = 0; k < n; k++) memcpy(&buf[k * w], &c->px[(y + k) * W + x0], (size_t)w * 2);
        esp_lcd_panel_draw_bitmap(d->panel, x0, y, x1, y + n, buf);
    }
    c->has_dirty = false;
}

/* Wait until both DMA buffers are idle (all queued transfers done). */
static void flush_wait(struct display *d)
{
    xSemaphoreTake(d->inflight, portMAX_DELAY);
    xSemaphoreTake(d->inflight, portMAX_DELAY);
    xSemaphoreGive(d->inflight);
    xSemaphoreGive(d->inflight);
}

/* ------------------------------------------------------------------ drawing (caller holds the lock) */
static void clear_bg(struct display *d) { canvas_fill_rect(&d->canvas, 0, 0, W, H, face_bg_rgb565()); }

/* Render face rect r (face coordinates) into the canvas and mark it dirty. */
static void face_render_rect(struct display *d, const face_frame_t *f, face_rect_t r)
{
    if (face_rect_empty(r)) return;
    face_sdf_render_rgb565be(f, r, d->canvas.px + OY * W + OX, W);
    canvas_dirty(&d->canvas, OX + r.x0, OY + r.y0, OX + r.x1 - 1, OY + r.y1 - 1);
}

/* One face frame: only the union of the previous and the new bounding box is rendered and sent. */
static void face_step(struct display *d, bool force)
{
    uint32_t now = now_ms32();
    uint32_t sid = 0, pos = 0;
    bool streaming = d->stream_fn && d->stream_fn(d->stream_ctx, &sid, &pos);
    face_engine_track_sync(&d->eng, now, streaming, sid, pos);
    face_frame_t f;
    face_engine_frame(&d->eng, now, &f);
    if (!force && memcmp(&f, &d->last_frame, sizeof(f)) == 0) return;
    face_rect_t nb = face_sdf_bbox(&f);
    face_rect_t r = force ? nb : face_rect_union(d->last_bbox, nb);
    if (force) clear_bg(d);
    face_render_rect(d, &f, r);
    d->last_frame = f;
    d->last_bbox = nb;
}

static void draw_text_screen(struct display *d)
{
    canvas_text_block(&d->canvas, d->text, d->text_n, d->text_size, d->text_fg, d->text_bg);
}

static void draw_status_screen(struct display *d)
{
    canvas_text_block(&d->canvas, d->status, (int)strlen(d->status), 2, 0xFFFF, 0x0000);
}

static void draw_image_screen(struct display *d)
{
    clear_bg(d);
    if (d->img_valid) canvas_blit_be(&d->canvas, OX, OY, FACE_PX, FACE_PX, d->img, FACE_PX);
}

static void set_mode_locked(struct display *d, display_mode_t m)
{
    if (m != d->mode) d->mode = m;
    d->mode_dirty = true;
    if (m == DISPLAY_MODE_CAMERA) d->cam_last_us = 0;
}

static void display_task(void *arg)
{
    struct display *d = arg;
    const TickType_t period = pdMS_TO_TICKS(1000 / CONFIG_ZANG_LCD_FPS);
    TickType_t wake = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&wake, period);
        xSemaphoreTake(d->lock, portMAX_DELAY);
        if (!d->paused) {
            bool force = d->mode_dirty;
            d->mode_dirty = false;
            if (d->mode == DISPLAY_MODE_TEXT && d->text_until_ms && (int64_t)now_ms32() >= d->text_until_ms) {
                set_mode_locked(d, DISPLAY_MODE_FACE);
                force = true;
                d->mode_dirty = false;
            }
            switch (d->mode) {
            case DISPLAY_MODE_FACE: face_step(d, force); break;
            case DISPLAY_MODE_IMAGE:
                if (force) draw_image_screen(d);
                break;
            case DISPLAY_MODE_TEXT:
                if (force) draw_text_screen(d);
                break;
            case DISPLAY_MODE_STATUS:
                if (force) draw_status_screen(d);
                break;
            case DISPLAY_MODE_CAMERA:
                if (force) clear_bg(d); /* frames arrive from the camera task */
                break;
            }
            flush_locked(d);
        }
        xSemaphoreGive(d->lock);
    }
}

/* ------------------------------------------------------------------ backlight / init */
static esp_err_t backlight_init(void)
{
    const ledc_timer_config_t t = {.speed_mode = LEDC_LOW_SPEED_MODE, .duty_resolution = LEDC_TIMER_10_BIT,
                                   .timer_num = BL_TIMER, .freq_hz = 20000, .clk_cfg = LEDC_AUTO_CLK};
    ESP_RETURN_ON_ERROR(ledc_timer_config(&t), TAG, "ledc timer");
    const ledc_channel_config_t ch = {
        .gpio_num = BOARD_PIN_LCD_BL, .speed_mode = LEDC_LOW_SPEED_MODE, .channel = BL_CHAN,
        .timer_sel = BL_TIMER, .duty = 0,
#if !CONFIG_ZANG_LCD_BL_ACTIVE_HIGH
        .flags.output_invert = 1, /* v0.0.1: P-MOSFET gate, LOW = on */
#endif
    };
    return ledc_channel_config(&ch);
}

esp_err_t display_set_backlight(display_handle_t d, uint8_t percent)
{
    if (BOARD_PIN_LCD_BL < 0) return ESP_ERR_NOT_SUPPORTED;
    if (percent > 100) percent = 100;
    ESP_RETURN_ON_ERROR(ledc_set_duty(LEDC_LOW_SPEED_MODE, BL_CHAN, 1023u * percent / 100u), TAG, "duty");
    esp_err_t err = ledc_update_duty(LEDC_LOW_SPEED_MODE, BL_CHAN);
    if (err == ESP_OK && d) d->backlight = percent;
    return err;
}

esp_err_t display_init(display_handle_t *out)
{
    ESP_RETURN_ON_FALSE(out, ESP_ERR_INVALID_ARG, TAG, "null");
    if (!BOARD_PINS_ASSIGNED(BOARD_PIN_LCD_MOSI, BOARD_PIN_LCD_SCLK, BOARD_PIN_LCD_DC)) {
        ESP_LOGW(TAG, "LCD pins unassigned: display disabled");
        return ESP_ERR_NOT_SUPPORTED;
    }
    struct display *d = calloc(1, sizeof(*d));
    ESP_RETURN_ON_FALSE(d, ESP_ERR_NO_MEM, TAG, "no mem");
    d->canvas.w = W;
    d->canvas.h = H;
    d->canvas.px = heap_caps_calloc((size_t)W * H, 2, MALLOC_CAP_SPIRAM);
    if (!d->canvas.px) d->canvas.px = heap_caps_calloc((size_t)W * H, 2, MALLOC_CAP_DEFAULT);
    d->img = heap_caps_calloc((size_t)FACE_PX * FACE_PX, 2, MALLOC_CAP_SPIRAM);
    d->cam = heap_caps_calloc((size_t)FACE_PX * FACE_PX, 2, MALLOC_CAP_SPIRAM);
    for (int i = 0; i < 2; i++) {
        d->line[i] = heap_caps_malloc((size_t)W * LINES * 2, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    }
    d->inflight = xSemaphoreCreateCounting(2, 2);
    d->lock = xSemaphoreCreateMutex();
    ESP_RETURN_ON_FALSE(d->canvas.px && d->line[0] && d->line[1] && d->inflight && d->lock, ESP_ERR_NO_MEM, TAG,
                        "buffers");
    if (!d->img || !d->cam) ESP_LOGW(TAG, "no PSRAM for image/camera buffers: image and preview modes disabled");

    const spi_bus_config_t bus = {.sclk_io_num = BOARD_PIN_LCD_SCLK, .mosi_io_num = BOARD_PIN_LCD_MOSI,
                                  .miso_io_num = -1, .quadwp_io_num = -1, .quadhd_io_num = -1,
                                  .max_transfer_sz = W * LINES * 2};
    ESP_RETURN_ON_ERROR(spi_bus_initialize(LCD_HOST, &bus, SPI_DMA_CH_AUTO), TAG, "spi bus");
    const esp_lcd_panel_io_spi_config_t io_cfg = {
        .dc_gpio_num = BOARD_PIN_LCD_DC, .cs_gpio_num = BOARD_PIN_LCD_CS,
        .pclk_hz = CONFIG_ZANG_LCD_PCLK_MHZ * 1000 * 1000, .lcd_cmd_bits = 8, .lcd_param_bits = 8,
        .spi_mode = CONFIG_ZANG_LCD_SPI_MODE, .trans_queue_depth = 4,
        .on_color_trans_done = on_trans_done, .user_ctx = d,
    };
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &d->io), TAG, "io");
    const esp_lcd_panel_dev_config_t pcfg = {
        .reset_gpio_num = BOARD_PIN_LCD_RST, /* -1: driver sends SWRESET (v0.0.1 has no reset line) */
#if CONFIG_ZANG_LCD_BGR
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_BGR,
#else
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB,
#endif
        .bits_per_pixel = 16,
    };
#if CONFIG_ZANG_LCD_ILI9341
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_ili9341(d->io, &pcfg, &d->panel), TAG, "ili9341");
#elif CONFIG_ZANG_LCD_GC9A01
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_gc9a01(d->io, &pcfg, &d->panel), TAG, "gc9a01");
#else
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_st7789(d->io, &pcfg, &d->panel), TAG, "st7789");
#endif
    ESP_RETURN_ON_ERROR(esp_lcd_panel_reset(d->panel), TAG, "reset");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_init(d->panel), TAG, "init");
    esp_lcd_panel_invert_color(d->panel, CONFIG_ZANG_LCD_INVERT_COLOR);
    esp_lcd_panel_set_gap(d->panel, CONFIG_ZANG_LCD_X_GAP, CONFIG_ZANG_LCD_Y_GAP);
    esp_lcd_panel_disp_on_off(d->panel, true);

    clear_bg(d);
    flush_locked(d);
    if (BOARD_PIN_LCD_BL >= 0 && backlight_init() == ESP_OK) display_set_backlight(d, 80);

    face_engine_init(&d->eng, now_ms32(), esp_random());
    d->mode = DISPLAY_MODE_FACE;
    d->mode_dirty = true;
    if (zang_task_create(ZANG_TASK_DISPLAY, display_task, d, NULL) != pdPASS) return ESP_ERR_NO_MEM;
    ESP_LOGI(TAG, "%dx%d %s @ %d MHz, %d-line DMA buffers x2, face %d emotions", W, H, PANEL_NAME,
             CONFIG_ZANG_LCD_PCLK_MHZ, LINES, FACE_N_EMOTIONS);
    *out = d;
    return ESP_OK;
}

void display_set_stream_source(display_handle_t d, display_stream_fn_t fn, void *ctx)
{
    if (!d) return;
    xSemaphoreTake(d->lock, portMAX_DELAY);
    d->stream_fn = fn, d->stream_ctx = ctx;
    xSemaphoreGive(d->lock);
}

/* ------------------------------------------------------------------ face API */
static void to_face_locked(struct display *d)
{
    if (d->mode != DISPLAY_MODE_FACE) set_mode_locked(d, DISPLAY_MODE_FACE);
}

esp_err_t display_face_set(display_handle_t d, uint8_t emotion, uint8_t intensity, uint16_t blend_ms, int8_t gaze_x,
                           int8_t gaze_y, uint8_t flags)
{
    ESP_RETURN_ON_FALSE(d, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(d->lock, portMAX_DELAY);
    face_engine_set(&d->eng, now_ms32(), emotion, intensity, blend_ms, gaze_x, gaze_y, flags);
    to_face_locked(d);
    xSemaphoreGive(d->lock);
    return ESP_OK;
}

esp_err_t display_face_params(display_handle_t d, uint16_t blend_ms, int n, const uint8_t *ids, const float *values)
{
    ESP_RETURN_ON_FALSE(d && ids && values, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(d->lock, portMAX_DELAY);
    face_engine_params(&d->eng, now_ms32(), blend_ms, n, ids, values);
    to_face_locked(d);
    xSemaphoreGive(d->lock);
    return ESP_OK;
}

esp_err_t display_face_track(display_handle_t d, uint32_t stream_id, const face_key_t *keys, int n)
{
    ESP_RETURN_ON_FALSE(d && keys, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(d->lock, portMAX_DELAY);
    face_engine_track_add(&d->eng, stream_id, keys, n); /* plays when the motion stream with this id runs */
    xSemaphoreGive(d->lock);
    return ESP_OK;
}

esp_err_t display_set_face(display_handle_t d, display_face_t face)
{
    ESP_RETURN_ON_FALSE(d && face < DISPLAY_FACE_COUNT, ESP_ERR_INVALID_ARG, TAG, "bad arg");
    static const uint8_t emo[DISPLAY_FACE_COUNT] = {
        [DISPLAY_FACE_NEUTRAL] = FACE_EMO_NEUTRAL, [DISPLAY_FACE_HAPPY] = FACE_EMO_HAPPY,
        [DISPLAY_FACE_SAD] = FACE_EMO_SAD, [DISPLAY_FACE_SURPRISED] = FACE_EMO_SURPRISED,
        [DISPLAY_FACE_SLEEPY] = FACE_EMO_SLEEPY, [DISPLAY_FACE_LISTENING] = FACE_EMO_LISTENING,
        [DISPLAY_FACE_THINKING] = FACE_EMO_THINKING, [DISPLAY_FACE_ERROR] = FACE_EMO_DIZZY,
    };
    esp_err_t err = display_face_set(d, emo[face], 255, 250, 0, 0, 0);
    if (err == ESP_OK && face == DISPLAY_FACE_ERROR) { /* X eyes in red */
        static const uint8_t ids[2] = {FP_HUE, FP_SAT};
        static const float vals[2] = {4.0f, 0.9f};
        err = display_face_params(d, 250, 2, ids, vals);
    }
    return err;
}

/* ------------------------------------------------------------------ other modes */
display_mode_t display_get_mode(display_handle_t d)
{
    if (!d) return DISPLAY_MODE_FACE;
    xSemaphoreTake(d->lock, portMAX_DELAY);
    display_mode_t m = d->mode;
    xSemaphoreGive(d->lock);
    return m;
}

uint8_t display_get_emotion(display_handle_t d)
{
    if (!d) return 0;
    xSemaphoreTake(d->lock, portMAX_DELAY);
    uint8_t e = d->eng.emotion;
    xSemaphoreGive(d->lock);
    return e;
}

esp_err_t display_show_text(display_handle_t d, uint8_t size, uint16_t fg, uint16_t bg, uint16_t duration_ms,
                            const char *text, int n)
{
    ESP_RETURN_ON_FALSE(d && text && n >= 0, ESP_ERR_INVALID_ARG, TAG, "bad arg");
    if (size < 1) size = 1;
    if (size > 4) size = 4;
    if (n > TEXT_MAX - 1) n = TEXT_MAX - 1;
    xSemaphoreTake(d->lock, portMAX_DELAY);
    memcpy(d->text, text, (size_t)n);
    d->text[n] = 0;
    d->text_n = n;
    d->text_size = size, d->text_fg = fg, d->text_bg = bg;
    d->text_until_ms = duration_ms ? (int64_t)now_ms32() + duration_ms : 0;
    set_mode_locked(d, DISPLAY_MODE_TEXT);
    xSemaphoreGive(d->lock);
    return ESP_OK;
}

esp_err_t display_set_status(display_handle_t d, const char *text)
{
    ESP_RETURN_ON_FALSE(d, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(d->lock, portMAX_DELAY);
    strlcpy(d->status, text ? text : "", sizeof(d->status));
    if (d->mode == DISPLAY_MODE_STATUS) d->mode_dirty = true;
    xSemaphoreGive(d->lock);
    return ESP_OK;
}

static esp_err_t load_sd_image(struct display *d, uint16_t id, uint16_t *dst)
{
    char path[48];
    snprintf(path, sizeof(path), DISPLAY_IMAGE_DIR "/%u.jpg", (unsigned)id);
    FILE *f = fopen(path, "rb");
    if (!f) return ESP_ERR_NOT_FOUND;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    esp_err_t err = ESP_ERR_INVALID_SIZE;
    if (len > 0 && len < 2 * 1024 * 1024) {
        uint8_t *buf = zjpeg_alloc_input((size_t)len);
        err = ESP_ERR_NO_MEM;
        if (buf) {
            err = fread(buf, 1, (size_t)len, f) == (size_t)len ? ESP_OK : ESP_FAIL;
            if (err == ESP_OK) err = zjpeg_decode_to_panel(buf, (size_t)len, IMG_FIT, face_bg_rgb565(), dst, FACE_PX, FACE_PX);
            zjpeg_free(buf);
        }
    }
    fclose(f);
    return err;
}

esp_err_t display_set_mode(display_handle_t d, display_mode_t mode, uint16_t image_id)
{
    ESP_RETURN_ON_FALSE(d && mode <= DISPLAY_MODE_STATUS, ESP_ERR_INVALID_ARG, TAG, "bad mode");
    esp_err_t err = ESP_OK;
    if (mode == DISPLAY_MODE_IMAGE) {
        ESP_RETURN_ON_FALSE(d->img, ESP_ERR_NO_MEM, TAG, "no image buffer");
        xSemaphoreTake(d->lock, portMAX_DELAY);
        bool cached = d->img_valid && d->img_id == image_id;
        xSemaphoreGive(d->lock);
        if (!cached) { /* decode the stored file outside the display lock, then publish */
            uint16_t *tmp = heap_caps_malloc((size_t)FACE_PX * FACE_PX * 2, MALLOC_CAP_SPIRAM);
            ESP_RETURN_ON_FALSE(tmp, ESP_ERR_NO_MEM, TAG, "no mem");
            err = load_sd_image(d, image_id, tmp);
            if (err == ESP_OK) {
                xSemaphoreTake(d->lock, portMAX_DELAY);
                memcpy(d->img, tmp, (size_t)FACE_PX * FACE_PX * 2);
                d->img_id = image_id, d->img_valid = true;
                xSemaphoreGive(d->lock);
            }
            heap_caps_free(tmp);
            if (err != ESP_OK) return err;
        }
    }
    xSemaphoreTake(d->lock, portMAX_DELAY);
    set_mode_locked(d, mode);
    xSemaphoreGive(d->lock);
    return err;
}

esp_err_t display_show_jpeg(display_handle_t d, uint16_t image_id, const uint8_t *jpg, size_t len)
{
    ESP_RETURN_ON_FALSE(d && jpg, ESP_ERR_INVALID_ARG, TAG, "null");
    ESP_RETURN_ON_FALSE(d->img, ESP_ERR_NO_MEM, TAG, "no image buffer");
    /* decode into a scratch buffer (not the one on screen), then copy under the lock */
    uint16_t *tmp = heap_caps_malloc((size_t)FACE_PX * FACE_PX * 2, MALLOC_CAP_SPIRAM);
    ESP_RETURN_ON_FALSE(tmp, ESP_ERR_NO_MEM, TAG, "no mem");
    esp_err_t err = zjpeg_decode_to_panel(jpg, len, IMG_FIT, face_bg_rgb565(), tmp, FACE_PX, FACE_PX);
    if (err == ESP_OK) {
        xSemaphoreTake(d->lock, portMAX_DELAY);
        memcpy(d->img, tmp, (size_t)FACE_PX * FACE_PX * 2);
        d->img_id = image_id, d->img_valid = true;
        set_mode_locked(d, DISPLAY_MODE_IMAGE);
        xSemaphoreGive(d->lock);
    }
    heap_caps_free(tmp);
    return err;
}

esp_err_t display_show_rgb565(display_handle_t d, uint16_t image_id, const uint16_t *px, size_t n_px)
{
    ESP_RETURN_ON_FALSE(d && px && n_px == (size_t)FACE_PX * FACE_PX, ESP_ERR_INVALID_SIZE, TAG, "need 240x240 px");
    ESP_RETURN_ON_FALSE(d->img, ESP_ERR_NO_MEM, TAG, "no image buffer");
    xSemaphoreTake(d->lock, portMAX_DELAY);
    for (size_t i = 0; i < n_px; i++) d->img[i] = rgb565_swap(px[i]);
    d->img_id = image_id, d->img_valid = true;
    set_mode_locked(d, DISPLAY_MODE_IMAGE);
    xSemaphoreGive(d->lock);
    return ESP_OK;
}

esp_err_t display_camera_frame(display_handle_t d, const uint8_t *jpg, size_t len)
{
    ESP_RETURN_ON_FALSE(d && jpg, ESP_ERR_INVALID_ARG, TAG, "null");
    if (!d->cam) return ESP_ERR_NO_MEM;
    xSemaphoreTake(d->lock, portMAX_DELAY);
    bool want = d->mode == DISPLAY_MODE_CAMERA && !d->paused;
    xSemaphoreGive(d->lock);
    if (!want) return ESP_ERR_INVALID_STATE;
    esp_err_t err = zjpeg_decode_to_panel(jpg, len, IMG_COVER, face_bg_rgb565(), d->cam, FACE_PX, FACE_PX);
    if (err != ESP_OK) return err;
    xSemaphoreTake(d->lock, portMAX_DELAY);
    if (d->mode == DISPLAY_MODE_CAMERA) canvas_blit_be(&d->canvas, OX, OY, FACE_PX, FACE_PX, d->cam, FACE_PX);
    xSemaphoreGive(d->lock);
    return ESP_OK;
}

esp_err_t display_blit_rgb565(display_handle_t d, int x, int y, int w, int h, const uint16_t *px)
{
    ESP_RETURN_ON_FALSE(d && px && w > 0 && h > 0, ESP_ERR_INVALID_ARG, TAG, "bad arg");
    xSemaphoreTake(d->lock, portMAX_DELAY);
    for (int yy = 0; yy < h; yy++) {
        int py = y + yy;
        if (py < 0 || py >= H) continue;
        for (int xx = 0; xx < w; xx++) {
            int pxx = x + xx;
            if (pxx < 0 || pxx >= W) continue;
            d->canvas.px[py * W + pxx] = rgb565_swap(px[yy * w + xx]);
        }
    }
    canvas_dirty(&d->canvas, x, y, x + w - 1, y + h - 1);
    xSemaphoreGive(d->lock);
    return ESP_OK;
}

esp_err_t display_store_jpeg(uint16_t image_id, const uint8_t *jpg, size_t len)
{
    ESP_RETURN_ON_FALSE(jpg && len, ESP_ERR_INVALID_ARG, TAG, "null");
    if (mkdir(DISPLAY_IMAGE_DIR, 0775) != 0 && errno != EEXIST) {
        ESP_LOGW(TAG, "mkdir %s failed (errno %d): SD card mounted?", DISPLAY_IMAGE_DIR, errno);
        return ESP_ERR_NOT_FOUND;
    }
    char path[48];
    snprintf(path, sizeof(path), DISPLAY_IMAGE_DIR "/%u.jpg", (unsigned)image_id);
    FILE *f = fopen(path, "wb");
    if (!f) return ESP_FAIL;
    size_t w = fwrite(jpg, 1, len, f);
    int cl = fclose(f);
    return (w == len && cl == 0) ? ESP_OK : ESP_FAIL;
}

/* ------------------------------------------------------------------ self test */
esp_err_t display_self_test(display_handle_t d, display_fps_t *out)
{
    ESP_RETURN_ON_FALSE(d, ESP_ERR_INVALID_ARG, TAG, "null");
    display_fps_t r = {0};
    xSemaphoreTake(d->lock, portMAX_DELAY);
    d->paused = true;
    const int n_full = 20, n_part = 60;
    static const uint16_t colors[3] = {0xF800, 0x07E0, 0x001F};
    int64_t t0 = esp_timer_get_time();
    for (int i = 0; i < n_full; i++) {
        canvas_fill_rect(&d->canvas, 0, 0, W, H, colors[i % 3]);
        flush_locked(d);
    }
    flush_wait(d);
    int64_t t1 = esp_timer_get_time();
    clear_bg(d);
    flush_locked(d);
    flush_wait(d);
    /* face frames: render + send the union of the old and new bounding box, as the animator does */
    face_frame_t f;
    memset(&f, 0, sizeof(f));
    memcpy(f.p, face_emotions[FACE_EMO_HAPPY].p, sizeof(f.p));
    face_rect_t prev = {0, 0, 0, 0};
    int64_t t2 = esp_timer_get_time();
    int64_t render_us = 0;
    for (int i = 0; i < n_part; i++) {
        f.blink = (i % 10) < 2 ? 0.05f : 1.0f;
        int64_t a = esp_timer_get_time();
        face_rect_t nb = face_sdf_bbox(&f);
        face_render_rect(d, &f, face_rect_union(prev, nb));
        render_us += esp_timer_get_time() - a;
        prev = nb;
        flush_locked(d);
    }
    flush_wait(d);
    int64_t t3 = esp_timer_get_time();
    r.full_fps = n_full * 1e6f / (float)(t1 - t0);
    r.partial_fps = n_part * 1e6f / (float)(t3 - t2);
    d->mode_dirty = true;
    d->paused = false;
    xSemaphoreGive(d->lock);
    ESP_LOGI(TAG, "self-test: full frame %.1f fps; face frame (render %.1f ms + send) %.1f fps (target %d)",
             r.full_fps, (float)render_us / 1000.0f / n_part, r.partial_fps, CONFIG_ZANG_LCD_FPS);
    if (out) *out = r;
    return ESP_OK;
}
