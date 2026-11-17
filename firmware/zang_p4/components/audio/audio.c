/* audio.c - NS4168 speaker over I2S (std, Philips, 16-bit stereo) + 2 PDM mics on I2S0 (PDM2PCM). */
#include "audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "board.h"
#include "driver/gpio.h"
#include "driver/i2s_pdm.h"
#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/stream_buffer.h"
#include "freertos/task.h"
#include "wav.h"
#include "zang_tasks.h"

static const char *TAG = "audio";

#define OUT_CHUNK      256      /* mono samples per I2S write */
#define MAX_RATE       48000
#define STREAM_GAP_MAX 16000    /* samples of silence inserted at most for a gap in a host stream */

typedef enum { SRC_NONE = 0, SRC_WAV, SRC_STREAM } src_t;

struct audio {
    i2s_chan_handle_t tx, rx;
    SemaphoreHandle_t lock;
    StreamBufferHandle_t sbuf; /* host stream PCM16 mono */
    /* --- protected by lock --- */
    src_t src;
    FILE *wav;
    wav_info_t wav_info;
    uint32_t wav_left;         /* bytes */
    uint32_t stream_id;
    uint32_t stream_next;      /* next expected sample offset into the stream buffer */
    bool stream_ended;
    uint32_t rate, pending_rate;
    uint64_t played;           /* samples played since the source started */
    int32_t gain_q15;
    /* mic */
    audio_mic_cb_t mic_cb;
    void *mic_ctx;
    TaskHandle_t mic_task;
};

/* ------------------------------------------------------------------ speaker */

static void stop_locked(struct audio *a)
{
    if (a->wav) {
        fclose(a->wav);
        a->wav = NULL;
    }
    a->src = SRC_NONE;
    xStreamBufferReset(a->sbuf);
}

/* Fill mono[] with up to n samples from the current source. Returns samples produced (0 = idle). */
static size_t pull_locked(struct audio *a, int16_t *mono, size_t n, bool *active)
{
    *active = a->src != SRC_NONE;
    if (a->src == SRC_WAV) {
        int16_t tmp[OUT_CHUNK * 2];
        size_t ch = a->wav_info.channels;
        size_t want = n * ch * 2;
        if (want > a->wav_left) want = a->wav_left - (a->wav_left % (ch * 2));
        size_t got = want ? fread(tmp, 1, want, a->wav) : 0;
        size_t frames = got / (ch * 2);
        for (size_t i = 0; i < frames; i++) {
            mono[i] = ch == 2 ? (int16_t)(((int32_t)tmp[2 * i] + tmp[2 * i + 1]) / 2) : tmp[i];
        }
        a->wav_left -= (uint32_t)got;
        if (frames == 0) stop_locked(a);
        return frames;
    }
    if (a->src == SRC_STREAM) {
        size_t got = xStreamBufferReceive(a->sbuf, mono, n * 2, 0) / 2;
        if (got < n) {
            if (a->stream_ended && got == 0) {
                stop_locked(a);
                return 0;
            }
            memset(&mono[got], 0, (n - got) * 2); /* underrun: keep the clock running with silence */
            got = n;
        }
        return got;
    }
    return 0;
}

static esp_err_t set_rate(struct audio *a, uint32_t rate)
{
    i2s_std_clk_config_t clk = I2S_STD_CLK_DEFAULT_CONFIG(rate);
    ESP_RETURN_ON_ERROR(i2s_channel_disable(a->tx), TAG, "disable");
    esp_err_t err = i2s_channel_reconfig_std_clock(a->tx, &clk);
    i2s_channel_enable(a->tx);
    if (err == ESP_OK) a->rate = rate;
    return err;
}

static void audio_out_task(void *arg)
{
    struct audio *a = arg;
    int16_t mono[OUT_CHUNK];
    int16_t stereo[OUT_CHUNK * 2];
    for (;;) {
        xSemaphoreTake(a->lock, portMAX_DELAY);
        if (a->pending_rate && a->pending_rate != a->rate) set_rate(a, a->pending_rate);
        a->pending_rate = 0;
        bool active;
        size_t n = pull_locked(a, mono, OUT_CHUNK, &active);
        if (n == 0) {
            n = OUT_CHUNK; /* idle: keep the amp fed with silence (no pops) */
            memset(mono, 0, sizeof(mono));
        } else if (active) {
            a->played += n;
        }
        int32_t g = a->gain_q15;
        xSemaphoreGive(a->lock);
        for (size_t i = 0; i < n; i++) {
            int16_t s = (int16_t)(((int32_t)mono[i] * g) >> 15);
            stereo[2 * i] = s; /* NS4168 plays the channel chosen by its CTRL pin; feed both */
            stereo[2 * i + 1] = s;
        }
        size_t written;
        i2s_channel_write(a->tx, stereo, n * 4, &written, portMAX_DELAY);
    }
}

static esp_err_t speaker_init(struct audio *a)
{
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(CONFIG_ZANG_AUDIO_OUT_I2S_PORT, I2S_ROLE_MASTER);
    cc.auto_clear = true;
    ESP_RETURN_ON_ERROR(i2s_new_channel(&cc, &a->tx, NULL), TAG, "tx channel");
    i2s_std_config_t std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(CONFIG_ZANG_AUDIO_OUT_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {.mclk = BOARD_PIN_I2S_MCLK, .bclk = BOARD_PIN_I2S_BCLK, .ws = BOARD_PIN_I2S_WS,
                     .dout = BOARD_PIN_I2S_DOUT, .din = I2S_GPIO_UNUSED},
    };
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(a->tx, &std), TAG, "std mode");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(a->tx), TAG, "enable");
    a->rate = CONFIG_ZANG_AUDIO_OUT_RATE;
    if (BOARD_PIN_AMP_EN >= 0) {
        gpio_config_t io = {.pin_bit_mask = BOARD_BIT(BOARD_PIN_AMP_EN), .mode = GPIO_MODE_OUTPUT};
        gpio_config(&io);
        gpio_set_level(BOARD_PIN_AMP_EN, 1);
    }
    size_t sz = (size_t)MAX_RATE * CONFIG_ZANG_AUDIO_OUT_BUF_MS / 1000 * 2;
    a->sbuf = xStreamBufferCreateWithCaps(sz, 2, MALLOC_CAP_SPIRAM);
    if (!a->sbuf) a->sbuf = xStreamBufferCreate(sz, 2);
    ESP_RETURN_ON_FALSE(a->sbuf, ESP_ERR_NO_MEM, TAG, "stream buffer");
    if (zang_task_create(ZANG_TASK_AUDIO_OUT, audio_out_task, a, NULL) != pdPASS) return ESP_ERR_NO_MEM;
    return ESP_OK;
}

/* ------------------------------------------------------------------ microphones */

static void audio_in_task(void *arg)
{
    struct audio *a = arg;
    const size_t frames = (size_t)CONFIG_ZANG_AUDIO_MIC_RATE * CONFIG_ZANG_AUDIO_MIC_FRAME_MS / 1000;
    int16_t *buf = heap_caps_malloc(frames * 2 * sizeof(int16_t), MALLOC_CAP_INTERNAL);
    if (!buf) {
        ESP_LOGE(TAG, "mic buffer");
        vTaskDelete(NULL);
        return;
    }
    for (;;) {
        size_t got = 0;
        if (i2s_channel_read(a->rx, buf, frames * 4, &got, portMAX_DELAY) != ESP_OK) continue;
        size_t n = got / 4; /* stereo frames */
#if !CONFIG_ZANG_AUDIO_MIC_STEREO
        for (size_t i = 0; i < n; i++) buf[i] = (int16_t)(((int32_t)buf[2 * i] + buf[2 * i + 1]) / 2);
        size_t samples = n;
#else
        size_t samples = n * 2;
#endif
        audio_mic_cb_t cb = a->mic_cb;
        if (cb) cb(buf, samples, CONFIG_ZANG_AUDIO_MIC_RATE, a->mic_ctx);
    }
}

static esp_err_t mic_init(struct audio *a)
{
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER); /* PDM RX lives on I2S0 */
    ESP_RETURN_ON_ERROR(i2s_new_channel(&cc, NULL, &a->rx), TAG, "rx channel");
    i2s_pdm_rx_config_t pdm = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(CONFIG_ZANG_AUDIO_MIC_RATE),
        .slot_cfg = I2S_PDM_RX_SLOT_PCM_FMT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {.clk = BOARD_PIN_PDM_CLK, .din = BOARD_PIN_PDM_DATA},
    };
    ESP_RETURN_ON_ERROR(i2s_channel_init_pdm_rx_mode(a->rx, &pdm), TAG, "pdm rx");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(a->rx), TAG, "enable");
    if (zang_task_create(ZANG_TASK_AUDIO_IN, audio_in_task, a, &a->mic_task) != pdPASS) return ESP_ERR_NO_MEM;
    return ESP_OK;
}

/* ------------------------------------------------------------------ API */

esp_err_t audio_init(audio_handle_t *out)
{
    ESP_RETURN_ON_FALSE(out, ESP_ERR_INVALID_ARG, TAG, "null");
    bool spk = BOARD_PINS_ASSIGNED(BOARD_PIN_I2S_BCLK, BOARD_PIN_I2S_WS, BOARD_PIN_I2S_DOUT);
    bool mic = BOARD_PINS_ASSIGNED(BOARD_PIN_PDM_CLK, BOARD_PIN_PDM_DATA);
    if (!spk && !mic) {
        ESP_LOGW(TAG, "no audio pins assigned: audio disabled");
        return ESP_ERR_NOT_SUPPORTED;
    }
    struct audio *a = calloc(1, sizeof(*a));
    ESP_RETURN_ON_FALSE(a, ESP_ERR_NO_MEM, TAG, "no mem");
    a->lock = xSemaphoreCreateMutex();
    a->gain_q15 = 32767 * 70 / 100;
    ESP_RETURN_ON_FALSE(a->lock, ESP_ERR_NO_MEM, TAG, "lock");
    if (spk) {
        esp_err_t err = speaker_init(a);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "speaker init failed: %s", esp_err_to_name(err));
            a->tx = NULL;
        }
    }
    if (mic) {
        esp_err_t err = mic_init(a);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "mic init failed: %s", esp_err_to_name(err));
            a->rx = NULL;
        }
    }
    ESP_LOGI(TAG, "speaker %s (I2S%d, %d Hz), mics %s (PDM %d Hz)", a->tx ? "ok" : "off",
             CONFIG_ZANG_AUDIO_OUT_I2S_PORT, CONFIG_ZANG_AUDIO_OUT_RATE, a->rx ? "ok" : "off",
             CONFIG_ZANG_AUDIO_MIC_RATE);
    *out = a;
    return ESP_OK;
}

bool audio_has_speaker(audio_handle_t a) { return a && a->tx; }
bool audio_has_mic(audio_handle_t a) { return a && a->rx; }

esp_err_t audio_set_volume(audio_handle_t a, uint8_t percent)
{
    ESP_RETURN_ON_FALSE(a, ESP_ERR_INVALID_ARG, TAG, "null");
    if (percent > 100) percent = 100;
    xSemaphoreTake(a->lock, portMAX_DELAY);
    a->gain_q15 = 32767 * percent / 100;
    xSemaphoreGive(a->lock);
    return ESP_OK;
}

esp_err_t audio_play_wav(audio_handle_t a, const char *path)
{
    ESP_RETURN_ON_FALSE(a && path, ESP_ERR_INVALID_ARG, TAG, "null");
    ESP_RETURN_ON_FALSE(a->tx, ESP_ERR_NOT_SUPPORTED, TAG, "no speaker");
    FILE *f = fopen(path, "rb");
    ESP_RETURN_ON_FALSE(f, ESP_ERR_NOT_FOUND, TAG, "cannot open %s", path);
    uint8_t hdr[512];
    size_t n = fread(hdr, 1, sizeof(hdr), f);
    wav_info_t info;
    if (wav_parse(hdr, n, &info) != 0 || info.sample_rate > MAX_RATE) {
        fclose(f);
        ESP_LOGE(TAG, "%s: not a 16-bit PCM WAV <= 48 kHz", path);
        return ESP_ERR_NOT_SUPPORTED;
    }
    fseek(f, (long)info.data_offset, SEEK_SET);
    xSemaphoreTake(a->lock, portMAX_DELAY);
    stop_locked(a);
    a->wav = f;
    a->wav_info = info;
    a->wav_left = info.data_bytes;
    a->pending_rate = info.sample_rate;
    a->played = 0;
    a->src = SRC_WAV;
    xSemaphoreGive(a->lock);
    return ESP_OK;
}

esp_err_t audio_stream_begin(audio_handle_t a, uint32_t stream_id, uint32_t sample_rate)
{
    ESP_RETURN_ON_FALSE(a && sample_rate && sample_rate <= MAX_RATE, ESP_ERR_INVALID_ARG, TAG, "bad arg");
    ESP_RETURN_ON_FALSE(a->tx, ESP_ERR_NOT_SUPPORTED, TAG, "no speaker");
    xSemaphoreTake(a->lock, portMAX_DELAY);
    stop_locked(a);
    a->stream_id = stream_id;
    a->stream_next = 0;
    a->stream_ended = false;
    a->pending_rate = sample_rate;
    a->played = 0;
    a->src = SRC_STREAM;
    xSemaphoreGive(a->lock);
    return ESP_OK;
}

esp_err_t audio_stream_write(audio_handle_t a, uint32_t stream_id, uint32_t sample_offset, const int16_t *pcm,
                             size_t n)
{
    ESP_RETURN_ON_FALSE(a && (pcm || !n), ESP_ERR_INVALID_ARG, TAG, "bad arg");
    xSemaphoreTake(a->lock, portMAX_DELAY);
    if (a->src != SRC_STREAM || stream_id != a->stream_id) {
        xSemaphoreGive(a->lock);
        return ESP_ERR_INVALID_STATE;
    }
    if (sample_offset < a->stream_next) { /* overlap / late: drop what was already queued */
        size_t skip = a->stream_next - sample_offset;
        if (skip >= n) {
            xSemaphoreGive(a->lock);
            return ESP_OK;
        }
        pcm += skip;
        n -= skip;
    } else if (sample_offset > a->stream_next) { /* gap: pad with silence */
        static const int16_t zeros[256];
        size_t gap = sample_offset - a->stream_next;
        if (gap > STREAM_GAP_MAX) gap = STREAM_GAP_MAX;
        while (gap) {
            size_t k = gap > 256 ? 256 : gap;
            xStreamBufferSend(a->sbuf, zeros, k * 2, 0);
            gap -= k;
        }
        a->stream_next = sample_offset;
    }
    xSemaphoreGive(a->lock);
    /* Blocks (outside the lock) while the buffer is full: natural back-pressure on the TCP reader. */
    size_t sent = xStreamBufferSend(a->sbuf, pcm, n * 2, pdMS_TO_TICKS(500)) / 2;
    xSemaphoreTake(a->lock, portMAX_DELAY);
    a->stream_next += (uint32_t)sent;
    xSemaphoreGive(a->lock);
    return sent == n ? ESP_OK : ESP_ERR_TIMEOUT;
}

esp_err_t audio_stream_end(audio_handle_t a, uint32_t stream_id)
{
    ESP_RETURN_ON_FALSE(a, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(a->lock, portMAX_DELAY);
    esp_err_t err = (a->src == SRC_STREAM && a->stream_id == stream_id) ? ESP_OK : ESP_ERR_INVALID_STATE;
    if (err == ESP_OK) a->stream_ended = true;
    xSemaphoreGive(a->lock);
    return err;
}

esp_err_t audio_stop(audio_handle_t a)
{
    ESP_RETURN_ON_FALSE(a, ESP_ERR_INVALID_ARG, TAG, "null");
    xSemaphoreTake(a->lock, portMAX_DELAY);
    stop_locked(a);
    xSemaphoreGive(a->lock);
    return ESP_OK;
}

bool audio_is_playing(audio_handle_t a)
{
    if (!a) return false;
    xSemaphoreTake(a->lock, portMAX_DELAY);
    bool p = a->src != SRC_NONE;
    xSemaphoreGive(a->lock);
    return p;
}

bool audio_clock_ms(audio_handle_t a, uint32_t *ms)
{
    if (!a || !ms) return false;
    xSemaphoreTake(a->lock, portMAX_DELAY);
    bool p = a->src != SRC_NONE && a->rate;
    if (p) *ms = (uint32_t)(a->played * 1000u / a->rate);
    xSemaphoreGive(a->lock);
    return p;
}

esp_err_t audio_mic_start(audio_handle_t a, audio_mic_cb_t cb, void *ctx)
{
    ESP_RETURN_ON_FALSE(a && cb, ESP_ERR_INVALID_ARG, TAG, "null");
    ESP_RETURN_ON_FALSE(a->rx, ESP_ERR_NOT_SUPPORTED, TAG, "no mic");
    a->mic_ctx = ctx;
    a->mic_cb = cb;
    return ESP_OK;
}

esp_err_t audio_mic_stop(audio_handle_t a)
{
    ESP_RETURN_ON_FALSE(a, ESP_ERR_INVALID_ARG, TAG, "null");
    a->mic_cb = NULL;
    return ESP_OK;
}
