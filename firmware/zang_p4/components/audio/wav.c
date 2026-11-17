/* wav.c - minimal RIFF/WAVE parser: walks chunks until "data". */
#include "wav.h"

#include <string.h>

static uint32_t le32(const uint8_t *p) { return (uint32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24)); }
static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

int wav_parse(const uint8_t *b, size_t n, wav_info_t *out)
{
    if (n < 12 || memcmp(b, "RIFF", 4) != 0 || memcmp(b + 8, "WAVE", 4) != 0) return -1;
    size_t off = 12;
    int have_fmt = 0;
    memset(out, 0, sizeof(*out));
    while (off + 8 <= n) {
        uint32_t sz = le32(b + off + 4);
        if (memcmp(b + off, "fmt ", 4) == 0) {
            if (off + 8 + 16 > n) return -2;
            uint16_t format = le16(b + off + 8);
            out->channels = le16(b + off + 10);
            out->sample_rate = le32(b + off + 12);
            out->bits = le16(b + off + 22);
            if ((format != 1 && format != 0xFFFE) || out->bits != 16 || out->channels < 1 || out->channels > 2) {
                return -3;
            }
            have_fmt = 1;
        } else if (memcmp(b + off, "data", 4) == 0) {
            if (!have_fmt) return -3;
            out->data_offset = (uint32_t)(off + 8);
            out->data_bytes = sz;
            return 0;
        }
        off += 8 + sz + (sz & 1);
    }
    return -2;
}
