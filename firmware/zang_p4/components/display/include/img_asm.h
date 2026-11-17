/* img_asm.h - reassembly of DISPLAY_IMAGE chunks (TCP delivers them in order). Pure C (no IDF), host-tested. */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IMG_ASM_MORE = 0,     /* chunk accepted, image incomplete */
    IMG_ASM_DONE = 1,     /* image complete: img_asm_data()/img_asm_len() are valid until the next feed */
    IMG_ASM_ERR_SIZE = -1,   /* total_len is 0 or larger than the buffer limit */
    IMG_ASM_ERR_ORDER = -2,  /* chunk offset is not the next expected byte (data lost): image abandoned */
    IMG_ASM_ERR_NOMEM = -3,
    IMG_ASM_ERR_RANGE = -4,  /* chunk runs past total_len */
} img_asm_status_t;

typedef struct {
    void *(*alloc)(size_t n); /* buffer allocator (cache-line aligned PSRAM on the robot, malloc on the host) */
    void (*release)(void *p);
    size_t max_bytes;
    /* state */
    uint8_t *buf;
    size_t cap, got, total;
    uint16_t id;
    uint8_t format, flags;
    bool active;
} img_asm_t;

void img_asm_init(img_asm_t *a, void *(*alloc)(size_t), void (*release)(void *), size_t max_bytes);
void img_asm_reset(img_asm_t *a);
/* A chunk with offset 0 starts a new image (dropping any partial one). Later chunks must continue it. */
img_asm_status_t img_asm_feed(img_asm_t *a, uint16_t id, uint8_t format, uint8_t flags, uint32_t total, uint32_t offset,
                              const uint8_t *data, uint16_t n);
static inline const uint8_t *img_asm_data(const img_asm_t *a) { return a->buf; }
static inline size_t img_asm_len(const img_asm_t *a) { return a->got; }

#ifdef __cplusplus
}
#endif
