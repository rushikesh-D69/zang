/* wav.h - minimal RIFF/WAVE PCM16 header parser (pure C). */
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t bits;
    uint32_t data_offset; /* file offset of the first sample */
    uint32_t data_bytes;
} wav_info_t;

/* Parse the first n bytes of a file. 0 = ok, <0 = not a PCM16 WAV / need more bytes. */
int wav_parse(const uint8_t *b, size_t n, wav_info_t *out);
