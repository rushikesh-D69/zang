/*
 * audio.h - speaker (I2S -> NS4168) and microphones (2 x PDM -> PDM2PCM on I2S0).
 * Playback: 16-bit PCM from WAV files on SD or streamed from the host (ZLink AUDIO_CHUNK).
 * audio_clock_ms() is the playback clock that ZLink streams with clock = 1 (audio) follow.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct audio *audio_handle_t;

/* Mic callback: n samples of 16-bit PCM (mono mix, or interleaved L/R if ZANG_AUDIO_MIC_STEREO). */
typedef void (*audio_mic_cb_t)(const int16_t *pcm, size_t n_samples, uint32_t sample_rate, void *ctx);

/* Each half is optional: ESP_ERR_NOT_SUPPORTED only if neither speaker nor mic pins are assigned. */
esp_err_t audio_init(audio_handle_t *out);
bool audio_has_speaker(audio_handle_t a);
bool audio_has_mic(audio_handle_t a);

esp_err_t audio_set_volume(audio_handle_t a, uint8_t percent);
/* Play a 16-bit PCM WAV (mono or stereo, any rate) from a path, asynchronously. Stops what is playing. */
esp_err_t audio_play_wav(audio_handle_t a, const char *path);
/* Host stream: begin at a sample rate, then write mono PCM16 in order. Gaps are filled with silence,
 * late (already played) samples are dropped, using sample_offset. */
esp_err_t audio_stream_begin(audio_handle_t a, uint32_t stream_id, uint32_t sample_rate);
esp_err_t audio_stream_write(audio_handle_t a, uint32_t stream_id, uint32_t sample_offset, const int16_t *pcm,
                             size_t n);
/* No more data will come: playback stops (and the clock ends) once the buffer drains. */
esp_err_t audio_stream_end(audio_handle_t a, uint32_t stream_id);
esp_err_t audio_stop(audio_handle_t a);
bool audio_is_playing(audio_handle_t a);
/* Milliseconds of audio played since the current WAV/stream started. false if nothing is playing. */
bool audio_clock_ms(audio_handle_t a, uint32_t *ms);

esp_err_t audio_mic_start(audio_handle_t a, audio_mic_cb_t cb, void *ctx);
esp_err_t audio_mic_stop(audio_handle_t a);

#ifdef __cplusplus
}
#endif
