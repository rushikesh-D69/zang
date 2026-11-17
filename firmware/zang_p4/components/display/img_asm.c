/* img_asm.c - see img_asm.h. */
#include "img_asm.h"

#include <string.h>

void img_asm_init(img_asm_t *a, void *(*alloc)(size_t), void (*release)(void *), size_t max_bytes)
{
    memset(a, 0, sizeof(*a));
    a->alloc = alloc, a->release = release, a->max_bytes = max_bytes;
}

void img_asm_reset(img_asm_t *a)
{
    a->active = false;
    a->got = a->total = 0;
}

img_asm_status_t img_asm_feed(img_asm_t *a, uint16_t id, uint8_t format, uint8_t flags, uint32_t total, uint32_t offset,
                              const uint8_t *data, uint16_t n)
{
    if (offset == 0) { /* start (or restart) */
        img_asm_reset(a);
        if (total == 0 || total > a->max_bytes) return IMG_ASM_ERR_SIZE;
        if (!a->buf || a->cap < total) {
            if (a->buf) a->release(a->buf);
            a->buf = a->alloc(total);
            a->cap = a->buf ? total : 0;
            if (!a->buf) return IMG_ASM_ERR_NOMEM;
        }
        a->id = id, a->format = format, a->flags = flags, a->total = total;
        a->active = true;
    } else if (!a->active || id != a->id || total != a->total) {
        img_asm_reset(a);
        return IMG_ASM_ERR_ORDER; /* a chunk of an image whose start we never saw */
    }
    if (offset > a->got) { /* hole */
        img_asm_reset(a);
        return IMG_ASM_ERR_ORDER;
    }
    if ((size_t)offset + n > a->total) {
        img_asm_reset(a);
        return IMG_ASM_ERR_RANGE;
    }
    if (offset < a->got) { /* retransmission of bytes we already have */
        size_t skip = a->got - offset;
        if (skip >= n) return a->got >= a->total ? IMG_ASM_DONE : IMG_ASM_MORE;
        data += skip, n = (uint16_t)(n - skip), offset += (uint32_t)skip;
    }
    memcpy(a->buf + offset, data, n);
    a->got = offset + n;
    if (a->got >= a->total) {
        a->active = false;
        return IMG_ASM_DONE;
    }
    return IMG_ASM_MORE;
}
