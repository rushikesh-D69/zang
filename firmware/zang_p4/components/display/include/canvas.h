/* canvas.h - RGB565 framebuffer in panel byte order with dirty-rect tracking, text, image scaling. Pure C (no IDF). */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t *px; /* w*h pixels, big-endian RGB565 as the panel expects */
    int w, h;
    bool has_dirty;
    int dx0, dy0, dx1, dy1; /* inclusive dirty rectangle */
} canvas_t;

static inline uint16_t rgb565_swap(uint16_t c) { return (uint16_t)((c << 8) | (c >> 8)); }

void canvas_dirty(canvas_t *c, int x0, int y0, int x1, int y1);
void canvas_fill_rect(canvas_t *c, int x, int y, int w, int h, uint16_t rgb565);
/* One line of text at scale 1..4 (8x8 font scaled by `scale`), top-left (x, y). Clipped. */
void canvas_text(canvas_t *c, int x, int y, int scale, const char *s, uint16_t fg, uint16_t bg);

/* Word-wrap: split s[0..n) into lines of at most `cols` glyphs (breaking at spaces, hard-breaking long words,
 * honouring '\n'). Non-printable and non-ASCII bytes count as one '?' glyph each (UTF-8 is not rendered).
 * Fills start/len of up to max_lines lines and returns the number of lines (truncated at max_lines). */
int text_wrap(const char *s, int n, int cols, int *start, int *len, int max_lines);
/* Wrapped text centred in the panel (both axes), on a bg-filled panel. Returns the number of lines drawn. */
int canvas_text_block(canvas_t *c, const char *s, int n, int scale, uint16_t fg, uint16_t bg);

/* Copy a w x h block of BIG-endian RGB565 (panel order) to (x, y). */
void canvas_blit_be(canvas_t *c, int x, int y, int w, int h, const uint16_t *px, int stride);

typedef enum {
    IMG_FIT = 0,     /* whole image visible, bars in bg colour (images) */
    IMG_COVER = 1,   /* centre crop that fills the destination (camera preview) */
    IMG_STRETCH = 2, /* ignore the aspect ratio */
} img_fit_t;

/* Nearest-neighbour scale of RGB565 (native byte order, as the JPEG decoder writes it; sw x sh visible pixels in a
 * buffer with row stride sstride) into dst (dw x dh, panel byte order, stride dw). Bars use bg (native RGB565). */
void img_scale_to_panel(const uint16_t *src, int sw, int sh, int sstride, uint16_t *dst, int dw, int dh,
                        img_fit_t mode, uint16_t bg);

#ifdef __cplusplus
}
#endif
