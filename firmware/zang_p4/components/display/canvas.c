/* canvas.c - framebuffer primitives. Pure C (no IDF). */
#include "canvas.h"

#include <string.h>
#include "font8x8_tr.h"

void canvas_dirty(canvas_t *c, int x0, int y0, int x1, int y1)
{
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > c->w - 1) x1 = c->w - 1;
    if (y1 > c->h - 1) y1 = c->h - 1;
    if (x1 < x0 || y1 < y0) return;
    if (!c->has_dirty) {
        c->dx0 = x0, c->dy0 = y0, c->dx1 = x1, c->dy1 = y1;
        c->has_dirty = true;
        return;
    }
    if (x0 < c->dx0) c->dx0 = x0;
    if (y0 < c->dy0) c->dy0 = y0;
    if (x1 > c->dx1) c->dx1 = x1;
    if (y1 > c->dy1) c->dy1 = y1;
}

void canvas_fill_rect(canvas_t *c, int x, int y, int w, int h, uint16_t rgb565)
{
    int x0 = x < 0 ? 0 : x, y0 = y < 0 ? 0 : y;
    int x1 = x + w > c->w ? c->w : x + w, y1 = y + h > c->h ? c->h : y + h;
    if (x1 <= x0 || y1 <= y0) return;
    uint16_t v = rgb565_swap(rgb565); /* SPI panels take big-endian RGB565 */
    for (int yy = y0; yy < y1; yy++) {
        uint16_t *row = &c->px[yy * c->w];
        for (int xx = x0; xx < x1; xx++) row[xx] = v;
    }
    canvas_dirty(c, x0, y0, x1 - 1, y1 - 1);
}

static unsigned glyph_of(unsigned char ch) { return (ch >= 0x20 && ch < 0x7F) ? ch : (unsigned)'?'; }

void canvas_text(canvas_t *c, int x, int y, int scale, const char *s, uint16_t fg, uint16_t bg)
{
    if (scale < 1) scale = 1;
    uint16_t f = rgb565_swap(fg), b = rgb565_swap(bg);
    int x_start = x;
    for (; *s; s++) {
        unsigned ch = glyph_of((unsigned char)*s);
        for (int col = 0; col < 8; col++) {
            uint8_t bits = font8x8_tr[ch][col];
            for (int row = 0; row < 8; row++) {
                uint16_t v = (bits >> row) & 1 ? f : b;
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        int px = x + col * scale + sx, py = y + row * scale + sy;
                        if (px >= 0 && px < c->w && py >= 0 && py < c->h) c->px[py * c->w + px] = v;
                    }
                }
            }
        }
        x += 8 * scale;
    }
    canvas_dirty(c, x_start, y, x - 1, y + 8 * scale - 1);
}

int text_wrap(const char *s, int n, int cols, int *start, int *len, int max_lines)
{
    if (cols < 1 || max_lines < 1) return 0;
    int lines = 0, i = 0;
    while (i < n && lines < max_lines) {
        while (i < n && s[i] == ' ' && lines > 0 && start[lines - 1] + len[lines - 1] == i) i++; /* trim wrap spaces */
        if (i >= n) break;
        int j = i, last_space = -1;
        while (j < n && s[j] != '\n' && j - i < cols) {
            if (s[j] == ' ') last_space = j;
            j++;
        }
        int end;
        if (j >= n || s[j] == '\n') {
            end = j; /* the rest fits, or a newline */
        } else if (s[j] == ' ') {
            end = j; /* the line is exactly full and the next char is a space */
        } else if (last_space > i) {
            end = last_space; /* break at the last space */
        } else {
            end = j; /* a word longer than the line: hard break */
        }
        start[lines] = i, len[lines] = end - i, lines++;
        i = end;
        if (i < n && s[i] == '\n') i++;
        else
            while (i < n && s[i] == ' ') i++;
    }
    return lines;
}

#define MAX_TEXT_LINES 32

int canvas_text_block(canvas_t *c, const char *s, int n, int scale, uint16_t fg, uint16_t bg)
{
    if (scale < 1) scale = 1;
    if (scale > 4) scale = 4;
    canvas_fill_rect(c, 0, 0, c->w, c->h, bg);
    const int margin = c->w / 12; /* the corners of a round window are not visible */
    int cols = (c->w - 2 * margin) / (8 * scale);
    int st[MAX_TEXT_LINES], ln[MAX_TEXT_LINES];
    int nl = text_wrap(s, n, cols, st, ln, MAX_TEXT_LINES);
    int line_h = 8 * scale + scale * 2;
    int max_rows = (c->h - 2 * margin) / line_h;
    if (nl > max_rows) nl = max_rows < 1 ? 1 : max_rows;
    int y = (c->h - nl * line_h + scale * 2) / 2;
    char buf[64];
    for (int i = 0; i < nl; i++) {
        int m = ln[i] < (int)sizeof(buf) - 1 ? ln[i] : (int)sizeof(buf) - 1;
        memcpy(buf, &s[st[i]], (size_t)m);
        buf[m] = 0;
        int x = (c->w - m * 8 * scale) / 2;
        canvas_text(c, x, y + i * line_h, scale, buf, fg, bg);
    }
    return nl;
}

void canvas_blit_be(canvas_t *c, int x, int y, int w, int h, const uint16_t *px, int stride)
{
    for (int yy = 0; yy < h; yy++) {
        int py = y + yy;
        if (py < 0 || py >= c->h) continue;
        int x0 = x < 0 ? -x : 0, x1 = x + w > c->w ? c->w - x : w;
        if (x1 > x0) memcpy(&c->px[py * c->w + x + x0], &px[yy * stride + x0], (size_t)(x1 - x0) * 2);
    }
    canvas_dirty(c, x, y, x + w - 1, y + h - 1);
}

void img_scale_to_panel(const uint16_t *src, int sw, int sh, int sstride, uint16_t *dst, int dw, int dh,
                        img_fit_t mode, uint16_t bg)
{
    if (sw < 1 || sh < 1 || dw < 1 || dh < 1) return;
    /* destination rectangle that the image occupies, and the source window it shows (16.16 fixed point) */
    int ox = 0, oy = 0, ow = dw, oh = dh;
    int64_t sx0 = 0, sy0 = 0, sww = sw, shh = sh; /* source window origin/size in 1/65536 px */
    sww <<= 16, shh <<= 16;
    if (mode == IMG_FIT) {
        if ((int64_t)sw * dh > (int64_t)sh * dw) { /* wider than the panel: bars above and below */
            oh = (int)(((int64_t)sh * dw + sw / 2) / sw);
            if (oh < 1) oh = 1;
            oy = (dh - oh) / 2;
        } else {
            ow = (int)(((int64_t)sw * dh + sh / 2) / sh);
            if (ow < 1) ow = 1;
            ox = (dw - ow) / 2;
        }
    } else if (mode == IMG_COVER) {
        if ((int64_t)sw * dh > (int64_t)sh * dw) { /* crop the sides */
            int64_t keep = ((int64_t)sh * dw << 16) / dh; /* source width that maps to dw */
            sx0 = (sww - keep) / 2, sww = keep;
        } else {
            int64_t keep = ((int64_t)sw * dh << 16) / dw;
            sy0 = (shh - keep) / 2, shh = keep;
        }
    }
    uint16_t bgs = rgb565_swap(bg);
    for (int y = 0; y < dh; y++) {
        uint16_t *row = &dst[y * dw];
        if (y < oy || y >= oy + oh) {
            for (int x = 0; x < dw; x++) row[x] = bgs;
            continue;
        }
        int sy = (int)((sy0 + (shh * (2 * (y - oy) + 1)) / (2 * oh)) >> 16);
        if (sy >= sh) sy = sh - 1;
        const uint16_t *srow = &src[sy * sstride];
        for (int x = 0; x < dw; x++) {
            if (x < ox || x >= ox + ow) {
                row[x] = bgs;
                continue;
            }
            int sx = (int)((sx0 + (sww * (2 * (x - ox) + 1)) / (2 * ow)) >> 16);
            if (sx >= sw) sx = sw - 1;
            row[x] = rgb565_swap(srow[sx]);
        }
    }
}
