/* app_media.c - ZLink display / face / camera / microphone handling.
 *
 *   FACE_SET, FACE_PARAMS, FACE_TRACK      -> display component (face_engine + face_sdf)
 *   DISPLAY_IMAGE (TCP)                    -> img_asm reassembly -> zjpeg hardware decode -> display; SD store
 *   DISPLAY_TEXT, DISPLAY_MODE             -> display component
 *   CMD CAMERA_STREAM / SNAPSHOT           -> camera component (OV2640 JPEG) -> CAMERA_FRAME chunks over UDP
 *   CMD MIC_STREAM                         -> audio component (PDM2PCM 16 kHz) -> MIC_CHUNK over UDP
 *   CMD SET_VOLUME / SET_BRIGHTNESS        -> audio / display
 */
#include <string.h>
#include "app_internal.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "img_asm.h"
#include "sdkconfig.h"
#include "zjpeg.h"

static const char *TAG = "media";

#ifndef CONFIG_ZANG_CAMERA_DEFAULT_QUALITY
#define CONFIG_ZANG_CAMERA_DEFAULT_QUALITY 60 /* camera component disabled: the value is never used */
#endif
#define IMAGE_MAX_BYTES (CONFIG_ZANG_IMAGE_MAX_KB * 1024)
#define PREVIEW_PERIOD_US 66000 /* 15 fps preview */

/* DISPLAY_IMAGE / IMAGE_DONE status */
enum { IMG_OK = 0, IMG_ERR_DECODE = 1, IMG_ERR_STORE = 2, IMG_ERR_DATA = 3, IMG_ERR_UNSUPPORTED = 4 };

static struct {
    app_ctx_t c;
    bool host_face;
    img_asm_t img;
    /* camera */
    SemaphoreHandle_t lock;
    bool cam_stream;
    camera_size_t cam_size;
    uint8_t cam_fps, cam_quality;
    bool snapshot;
    int warmup;            /* frames to skip after a cold start for a snapshot (auto exposure) */
    bool cam_running;
    int64_t next_send_us, next_preview_us;
    uint32_t frame_id;
    esp_timer_handle_t cam_timer;
    uint8_t pkt[ZLINK_MAX_PAYLOAD];
    /* mic */
    bool mic_on;
} m;

bool app_host_owns_face(void) { return m.host_face; }

uint8_t app_media_display_mode(void) { return m.c.display ? (uint8_t)display_get_mode(m.c.display) : 0; }
uint8_t app_media_emotion(void) { return m.c.display ? display_get_emotion(m.c.display) : 0; }

/* ------------------------------------------------------------------ camera */
static bool preview_wanted(void) { return m.c.display && display_get_mode(m.c.display) == DISPLAY_MODE_CAMERA; }

static void cam_send_frame(const camera_frame_t *f)
{
    if (!app_has_peer()) return;
    uint32_t id = ++m.frame_id, ms = (uint32_t)(esp_timer_get_time() / 1000);
    for (uint32_t off = 0; off < f->len; off += ZLINK_CAMERA_CHUNK_MAX) {
        uint32_t n = f->len - off;
        if (n > ZLINK_CAMERA_CHUNK_MAX) n = ZLINK_CAMERA_CHUNK_MAX;
        zlink_camera_frame_t cf = {.frame_id = id, .robot_ms = ms, .w = (uint16_t)f->width, .h = (uint16_t)f->height,
                                   .format = 1, .total_len = (uint32_t)f->len, .offset = off,
                                   .data = f->data + off, .n_data = (uint16_t)n};
        uint16_t len;
        if (zlink_pack_camera_frame(&cf, m.pkt, sizeof(m.pkt), &len) != ZLINK_OK) return;
        app_send(ZLINK_VIA_UDP, ZLINK_CAMERA_FRAME, m.pkt, len);
        if ((off / ZLINK_CAMERA_CHUNK_MAX) % 6 == 5) vTaskDelay(1); /* let the Wi-Fi link drain */
    }
}

static void cam_reconcile(void); /* camera/preview/stream state -> camera hardware (not from the camera callback) */

static void cam_timer_cb(void *arg) { cam_reconcile(); }

/* Runs on the camera task: the buffer is only valid here. Never call camera_stop() from here. */
static void on_cam_frame(const camera_frame_t *f, void *ctx)
{
    int64_t now = f->t_us;
    xSemaphoreTake(m.lock, portMAX_DELAY);
    bool stream = m.cam_stream, snap = m.snapshot, prev = preview_wanted();
    int64_t period = stream && m.cam_fps ? 1000000 / m.cam_fps : 0;
    bool send = false;
    if (snap && m.warmup > 0) {
        m.warmup--;
        snap = false;
    }
    if (snap) {
        send = true;
        m.snapshot = false;
    } else if (stream && now >= m.next_send_us) {
        send = true;
        m.next_send_us = (m.next_send_us && now - m.next_send_us < period) ? m.next_send_us + period : now + period;
    }
    bool draw = prev && now >= m.next_preview_us;
    if (draw) m.next_preview_us = now + PREVIEW_PERIOD_US;
    bool idle = !m.cam_stream && !m.snapshot && !prev;
    xSemaphoreGive(m.lock);
    if (send) cam_send_frame(f);
    if (draw) display_camera_frame(m.c.display, f->data, f->len);
    if (idle) esp_timer_start_once(m.cam_timer, 1000); /* stop the sensor from the timer task */
}

static void cam_reconcile(void)
{
    if (!m.c.camera) return;
    xSemaphoreTake(m.lock, portMAX_DELAY);
    bool want = m.cam_stream || m.snapshot || preview_wanted();
    camera_size_t size = m.cam_stream ? m.cam_size : CAMERA_SIZE_QVGA;
    uint8_t q = m.cam_stream ? m.cam_quality : CONFIG_ZANG_CAMERA_DEFAULT_QUALITY;
    bool running = m.cam_running;
    xSemaphoreGive(m.lock);
    if (want) {
        esp_err_t err = camera_configure(m.c.camera, size, q);
        if (err == ESP_OK && !running) {
            err = camera_start(m.c.camera, on_cam_frame, NULL);
            xSemaphoreTake(m.lock, portMAX_DELAY);
            m.cam_running = err == ESP_OK;
            m.next_send_us = m.next_preview_us = 0;
            xSemaphoreGive(m.lock);
        }
        if (err != ESP_OK) ESP_LOGW(TAG, "camera: %s", esp_err_to_name(err));
    } else if (running) {
        camera_stop(m.c.camera);
        xSemaphoreTake(m.lock, portMAX_DELAY);
        m.cam_running = false;
        xSemaphoreGive(m.lock);
    }
}

static esp_err_t cmd_camera_stream(const zlink_cmd_t *cmd)
{
    zlink_camera_stream_t cs;
    if (zlink_cmd_camera_stream(cmd, &cs) != ZLINK_OK || cs.size > 1) return ESP_ERR_INVALID_ARG;
    if (!m.c.camera) return ESP_ERR_NOT_SUPPORTED;
    xSemaphoreTake(m.lock, portMAX_DELAY);
    m.cam_stream = cs.on != 0;
    m.cam_size = cs.size ? CAMERA_SIZE_VGA : CAMERA_SIZE_QVGA;
    m.cam_fps = cs.fps ? cs.fps : 10;
    if (m.cam_fps > 30) m.cam_fps = 30;
    m.cam_quality = cs.jpeg_quality ? cs.jpeg_quality : CONFIG_ZANG_CAMERA_DEFAULT_QUALITY;
    xSemaphoreGive(m.lock);
    cam_reconcile();
    return ESP_OK;
}

static esp_err_t cmd_snapshot(void)
{
    if (!m.c.camera) return ESP_ERR_NOT_SUPPORTED;
    xSemaphoreTake(m.lock, portMAX_DELAY);
    m.snapshot = true;
    m.warmup = m.cam_running ? 0 : 4; /* a cold sensor needs a few frames to settle exposure */
    xSemaphoreGive(m.lock);
    cam_reconcile();
    return ESP_OK;
}

/* ------------------------------------------------------------------ microphones */
static void on_mic(const int16_t *pcm, size_t n, uint32_t rate, void *ctx)
{
    static uint8_t p[ZLINK_MAX_PAYLOAD];
    uint16_t len;
    if (n * 2 > ZLINK_MAX_PAYLOAD - 8) n = (ZLINK_MAX_PAYLOAD - 8) / 2;
    zlink_mic_chunk_t mc = {.robot_ms = (uint32_t)(esp_timer_get_time() / 1000),
                            .codec = rate == 24000 ? ZLINK_CODEC_PCM16_24K : ZLINK_CODEC_PCM16_16K,
                            .data = (const uint8_t *)pcm, .n_data = (uint16_t)(n * 2)};
    if (zlink_pack_mic_chunk(&mc, p, sizeof(p), &len) == ZLINK_OK) app_send(ZLINK_VIA_UDP, ZLINK_MIC_CHUNK, p, len);
}

static esp_err_t cmd_mic_stream(const zlink_cmd_t *cmd)
{
    zlink_mic_stream_t ms;
    if (zlink_cmd_mic_stream(cmd, &ms) != ZLINK_OK) return ESP_ERR_INVALID_ARG;
    if (!m.c.audio || !audio_has_mic(m.c.audio)) return ESP_ERR_NOT_SUPPORTED;
    if (!ms.on) {
        m.mic_on = false;
        return audio_mic_stop(m.c.audio);
    }
#if CONFIG_ZANG_AUDIO_MIC_STEREO
    return ESP_ERR_NOT_SUPPORTED; /* MIC_CHUNK carries mono PCM */
#else
    if (ms.codec != ZLINK_CODEC_PCM16_16K) return ESP_ERR_NOT_SUPPORTED; /* only 16 kHz PCM; no 24 kHz, no Opus */
    esp_err_t err = audio_mic_start(m.c.audio, on_mic, NULL);
    m.mic_on = err == ESP_OK;
    return err;
#endif
}

esp_err_t app_media_cmd(const zlink_cmd_t *cmd)
{
    switch (cmd->cmd) {
    case ZLINK_CMD_MIC_STREAM: return cmd_mic_stream(cmd);
    case ZLINK_CMD_CAMERA_STREAM: return cmd_camera_stream(cmd);
    case ZLINK_CMD_SNAPSHOT: return cmd_snapshot();
    case ZLINK_CMD_SET_VOLUME: {
        uint8_t pct;
        if (zlink_cmd_percent(cmd, &pct) != ZLINK_OK || pct > 100) return ESP_ERR_INVALID_ARG;
        return m.c.audio ? audio_set_volume(m.c.audio, pct) : ESP_ERR_NOT_SUPPORTED;
    }
    case ZLINK_CMD_SET_BRIGHTNESS: {
        uint8_t pct;
        if (zlink_cmd_percent(cmd, &pct) != ZLINK_OK || pct > 100) return ESP_ERR_INVALID_ARG;
        return m.c.display ? display_set_backlight(m.c.display, pct) : ESP_ERR_NOT_SUPPORTED;
    }
    default: return ESP_ERR_NOT_FOUND;
    }
}

/* ------------------------------------------------------------------ DISPLAY_IMAGE */
static void image_done(uint16_t id, uint8_t status, uint16_t seq, zlink_via_t via)
{
    uint8_t ev[3] = {(uint8_t)(id & 0xFF), (uint8_t)(id >> 8), status};
    app_send_event(ZLINK_EV_IMAGE_DONE, ev, sizeof(ev));
    uint8_t out[8];
    uint16_t len;
    zlink_ack_t ack = {.seq_acked = seq, .status = status == IMG_OK ? 0 : (status == IMG_ERR_UNSUPPORTED ? 4 : 5)};
    if (zlink_pack_ack(&ack, out, sizeof(out), &len) != ZLINK_OK) return;
    app_send(via, ZLINK_ACK, out, len);
    if (via != ZLINK_VIA_UDP) app_send(ZLINK_VIA_UDP, ZLINK_ACK, out, len); /* clients that only read UDP */
}

static void on_display_image(const zlink_frame_t *f, zlink_via_t via)
{
    zlink_display_image_t im;
    if (zlink_unpack_display_image(f->payload, f->len, &im) != ZLINK_OK) return;
    bool was_active = m.img.active;
    img_asm_status_t st = img_asm_feed(&m.img, im.image_id, im.format, im.flags, im.total_len, im.offset, im.data,
                                       im.n_data);
    if (st == IMG_ASM_MORE) return;
    if (st != IMG_ASM_DONE) {
        ESP_LOGW(TAG, "image %u: bad chunk (%d)", im.image_id, (int)st);
        if (was_active || im.offset == 0) image_done(im.image_id, IMG_ERR_DATA, f->seq, via);
        return;
    }
    const uint8_t *data = img_asm_data(&m.img);
    size_t len = img_asm_len(&m.img);
    uint8_t status = IMG_OK;
    if (m.img.format == 1) {
        if ((m.img.flags & ZLINK_IMAGE_SHOW) && (!m.c.display || display_show_jpeg(m.c.display, m.img.id, data, len) != ESP_OK)) {
            status = m.c.display ? IMG_ERR_DECODE : IMG_ERR_UNSUPPORTED;
        }
        if ((m.img.flags & ZLINK_IMAGE_STORE) && display_store_jpeg(m.img.id, data, len) != ESP_OK && status == IMG_OK) {
            status = IMG_ERR_STORE;
        }
    } else if (m.img.format == 0) { /* raw RGB565 LE, 240 x 240 */
        if (len != 240u * 240u * 2u) {
            status = IMG_ERR_DATA;
        } else if (m.img.flags & ZLINK_IMAGE_SHOW) {
            if (!m.c.display || display_show_rgb565(m.c.display, m.img.id, (const uint16_t *)data, len / 2) != ESP_OK)
                status = m.c.display ? IMG_ERR_DECODE : IMG_ERR_UNSUPPORTED;
        }
        if ((m.img.flags & ZLINK_IMAGE_STORE) && status == IMG_OK) status = IMG_ERR_UNSUPPORTED; /* SD holds .jpg only */
    } else {
        status = IMG_ERR_UNSUPPORTED;
    }
    ESP_LOGI(TAG, "image %u: %u bytes, format %u, flags %u -> status %u", m.img.id, (unsigned)len, m.img.format,
             m.img.flags, status);
    image_done(m.img.id, status, f->seq, via);
    cam_reconcile(); /* showing an image leaves camera-preview mode */
}

/* ------------------------------------------------------------------ face / display messages */
bool app_media_on_frame(const zlink_frame_t *f, zlink_via_t via)
{
    display_handle_t d = m.c.display;
    switch (f->type) {
    case ZLINK_FACE_SET: {
        zlink_face_set_t fs;
        if (zlink_unpack_face_set(f->payload, f->len, &fs) != ZLINK_OK) break;
        if (fs.emotion >= FACE_N_EMOTIONS) {
            ESP_LOGW(TAG, "FACE_SET: unknown emotion %u", fs.emotion);
            break;
        }
        m.host_face = true;
        if (d) display_face_set(d, fs.emotion, fs.intensity, fs.blend_ms, fs.gaze_x, fs.gaze_y, fs.flags);
        cam_reconcile();
        break;
    }
    case ZLINK_FACE_PARAMS: {
        static zlink_face_params_t fp;
        if (zlink_unpack_face_params(f->payload, f->len, &fp) != ZLINK_OK) break;
        uint8_t ids[ZLINK_FACE_MAX_PARAMS];
        float vals[ZLINK_FACE_MAX_PARAMS];
        for (int i = 0; i < fp.n; i++) ids[i] = fp.params[i].id, vals[i] = fp.params[i].value;
        m.host_face = true;
        if (d) display_face_params(d, fp.blend_ms, fp.n, ids, vals);
        cam_reconcile();
        break;
    }
    case ZLINK_FACE_TRACK: {
        static zlink_face_track_t ft;
        static face_key_t keys[ZLINK_FACE_MAX_KEYS];
        if (zlink_unpack_face_track(f->payload, f->len, &ft) != ZLINK_OK) break;
        for (int i = 0; i < ft.n; i++) {
            keys[i] = (face_key_t){.t_ms = ft.keys[i].t_ms, .emotion = ft.keys[i].emotion,
                                   .intensity = ft.keys[i].intensity, .blend_ms = ft.keys[i].blend_ms};
        }
        m.host_face = true;
        if (d) display_face_track(d, ft.stream_id, keys, ft.n);
        break;
    }
    case ZLINK_DISPLAY_IMAGE:
        on_display_image(f, via);
        break;
    case ZLINK_DISPLAY_TEXT: {
        zlink_display_text_t t;
        if (zlink_unpack_display_text(f->payload, f->len, &t) != ZLINK_OK) break;
        if (d) display_show_text(d, t.size, t.fg, t.bg, t.duration_ms, t.text, t.n_text);
        cam_reconcile();
        break;
    }
    case ZLINK_DISPLAY_MODE: {
        zlink_display_mode_msg_t dm;
        if (zlink_unpack_display_mode_msg(f->payload, f->len, &dm) != ZLINK_OK || dm.mode > ZLINK_DISPLAY_STATUS) break;
        if (d) {
            esp_err_t err = display_set_mode(d, (display_mode_t)dm.mode, dm.image_id);
            if (err != ESP_OK) ESP_LOGW(TAG, "DISPLAY_MODE %u (image %u): %s", dm.mode, dm.image_id, esp_err_to_name(err));
        }
        cam_reconcile();
        break;
    }
    default:
        return false;
    }
    return true;
}

void app_media_start(const app_ctx_t *ctx)
{
    m.c = *ctx;
    m.lock = xSemaphoreCreateMutex();
    const esp_timer_create_args_t ta = {.callback = cam_timer_cb, .name = "cam_reconcile"};
    esp_timer_create(&ta, &m.cam_timer);
    img_asm_init(&m.img, zjpeg_alloc_input, zjpeg_free, IMAGE_MAX_BYTES);
    m.cam_fps = 10;
    m.cam_quality = CONFIG_ZANG_CAMERA_DEFAULT_QUALITY;
#if CONFIG_ZANG_STREAM_MIC
    if (m.c.audio && audio_has_mic(m.c.audio)) {
        m.mic_on = audio_mic_start(m.c.audio, on_mic, NULL) == ESP_OK;
    }
#endif
}
