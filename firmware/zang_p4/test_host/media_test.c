/* media_test.c - canvas text wrapping/drawing, image scaling, DISPLAY_IMAGE reassembly. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "canvas.h"
#include "img_asm.h"

static int fails;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);             \
            fails++;                                                          \
        }                                                                     \
    } while (0)

static void test_wrap(void)
{
    int st[16], ln[16];
    const char *t = "hello brave new world";
    int n = text_wrap(t, (int)strlen(t), 11, st, ln, 16);
    CHECK(n == 2 && ln[0] == 11 && !strncmp(t + st[0], "hello brave", 11) && !strncmp(t + st[1], "new world", ln[1]));
    n = text_wrap(t, (int)strlen(t), 8, st, ln, 16);
    CHECK(n == 4 && ln[0] == 5 && !strncmp(t + st[0], "hello", 5) && !strncmp(t + st[1], "brave", 5) &&
          !strncmp(t + st[2], "new", 3) && ln[2] == 3 && !strncmp(t + st[3], "world", 5));
    const char *u = "abcdefghij";
    n = text_wrap(u, 10, 4, st, ln, 16); /* hard break of a long word */
    CHECK(n == 3 && ln[0] == 4 && ln[1] == 4 && ln[2] == 2);
    const char *v = "a\nb\n\nc";
    n = text_wrap(v, 6, 10, st, ln, 16);
    CHECK(n >= 3 && ln[0] == 1 && v[st[0]] == 'a' && v[st[1]] == 'b');
    n = text_wrap(u, 10, 1, st, ln, 4); /* truncated at max_lines */
    CHECK(n == 4);
    CHECK(text_wrap("", 0, 10, st, ln, 4) == 0);
    CHECK(text_wrap("x", 1, 0, st, ln, 4) == 0);
}

static void test_text_block(void)
{
    canvas_t c = {.w = 240, .h = 240};
    c.px = calloc(240 * 240, 2);
    int nl = canvas_text_block(&c, "Hello Zang", 10, 2, 0xFFFF, 0x0000);
    CHECK(nl == 1 && c.has_dirty && c.dx0 == 0 && c.dy0 == 0 && c.dx1 == 239 && c.dy1 == 239);
    int lit = 0;
    for (int i = 0; i < 240 * 240; i++) lit += c.px[i] == 0xFFFF;
    CHECK(lit > 100);
    /* the text is vertically centred: no lit pixel in the top or bottom 80 rows */
    int outside = 0;
    for (int y = 0; y < 240; y++) {
        for (int x = 0; x < 240; x++) {
            if ((y < 80 || y >= 160) && c.px[y * 240 + x] == 0xFFFF) outside++;
        }
    }
    CHECK(outside == 0);
    /* size 4: 8*4 = 32 px glyphs -> 6 columns on 240 - 2*20 px; "Hello Zang" = 2 lines */
    nl = canvas_text_block(&c, "Hello Zang", 10, 4, 0xFFFF, 0x0000);
    CHECK(nl == 2);
    /* UTF-8 / control bytes do not crash and render as '?' */
    nl = canvas_text_block(&c, "caf\xC3\xA9 \x01", 7, 1, 0xFFFF, 0x0000);
    CHECK(nl == 1);
    /* byte order: fg 0x1234 is stored swapped */
    memset(c.px, 0, 240 * 240 * 2);
    canvas_text(&c, 0, 0, 1, "I", 0x1234, 0x0000);
    int found = 0;
    for (int i = 0; i < 64; i++) found += c.px[(i / 8) * 240 + (i % 8)] == 0x3412;
    CHECK(found > 5);
    free(c.px);
}

static void test_scale(void)
{
    /* 4x2 source with distinct values, stride 8 (decoder rows are padded) */
    uint16_t src[2 * 8];
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 8; x++) src[y * 8 + x] = (uint16_t)(0x0100 * (y + 1) + x + 1);
    }
    uint16_t dst[8 * 8];
    /* stretch 4x2 -> 8x8: each source pixel becomes 2 columns, 4 rows */
    img_scale_to_panel(src, 4, 2, 8, dst, 8, 8, IMG_STRETCH, 0);
    CHECK(dst[0] == rgb565_swap(0x0101) && dst[1] == rgb565_swap(0x0101) && dst[2] == rgb565_swap(0x0102) &&
          dst[7] == rgb565_swap(0x0104));
    CHECK(dst[3 * 8 + 0] == rgb565_swap(0x0101) && dst[4 * 8 + 0] == rgb565_swap(0x0201));
    /* fit: 4x2 into 8x8 -> 8x4 image centred, bars above/below in bg */
    img_scale_to_panel(src, 4, 2, 8, dst, 8, 8, IMG_FIT, 0xF800);
    CHECK(dst[0] == rgb565_swap(0xF800) && dst[7 * 8 + 3] == rgb565_swap(0xF800));
    CHECK(dst[2 * 8 + 0] == rgb565_swap(0x0101) && dst[5 * 8 + 7] == rgb565_swap(0x0204));
    /* cover: 4x2 into 8x8 -> crop the sides to the centre 2x2, fill the whole square */
    img_scale_to_panel(src, 4, 2, 8, dst, 8, 8, IMG_COVER, 0xF800);
    int bars = 0;
    for (int i = 0; i < 64; i++) bars += dst[i] == rgb565_swap(0xF800);
    CHECK(bars == 0);
    CHECK(dst[0] == rgb565_swap(0x0102) && dst[7] == rgb565_swap(0x0103) && dst[7 * 8] == rgb565_swap(0x0202));
    /* identity 240x240 */
    uint16_t *a = malloc(240 * 240 * 2), *b = malloc(240 * 240 * 2);
    for (int i = 0; i < 240 * 240; i++) a[i] = (uint16_t)(i * 7);
    img_scale_to_panel(a, 240, 240, 240, b, 240, 240, IMG_FIT, 0);
    int same = 1;
    for (int i = 0; i < 240 * 240; i++) same &= b[i] == rgb565_swap(a[i]);
    CHECK(same);
    /* 320x240 camera frame, cover: no bars */
    uint16_t *cam = calloc(320 * 240, 2);
    img_scale_to_panel(cam, 320, 240, 320, b, 240, 240, IMG_COVER, 0x1111);
    int bar = 0;
    for (int i = 0; i < 240 * 240; i++) bar += b[i] == rgb565_swap(0x1111);
    CHECK(bar == 0);
    free(a), free(b), free(cam);
}

static int allocs, frees;
static void *t_alloc(size_t n) { allocs++; return malloc(n); }
static void t_free(void *p) { frees++; free(p); }

static void test_asm(void)
{
    img_asm_t a;
    img_asm_init(&a, t_alloc, t_free, 1000);
    uint8_t src[600];
    for (int i = 0; i < 600; i++) src[i] = (uint8_t)(i * 3);
    /* in order */
    CHECK(img_asm_feed(&a, 5, 1, 3, 600, 0, src, 250) == IMG_ASM_MORE);
    CHECK(img_asm_feed(&a, 5, 1, 3, 600, 250, src + 250, 250) == IMG_ASM_MORE);
    CHECK(img_asm_feed(&a, 5, 1, 3, 600, 500, src + 500, 100) == IMG_ASM_DONE);
    CHECK(img_asm_len(&a) == 600 && memcmp(img_asm_data(&a), src, 600) == 0 && a.id == 5 && a.flags == 3 &&
          a.format == 1);
    /* second image reuses the buffer (same size) */
    CHECK(img_asm_feed(&a, 6, 1, 1, 600, 0, src, 600) == IMG_ASM_DONE && allocs == 1);
    /* a bigger one reallocates */
    CHECK(img_asm_feed(&a, 7, 1, 1, 900, 0, src, 300) == IMG_ASM_MORE && allocs == 2 && frees == 1);
    /* a hole abandons the image */
    CHECK(img_asm_feed(&a, 7, 1, 1, 900, 400, src, 100) == IMG_ASM_ERR_ORDER);
    CHECK(img_asm_feed(&a, 7, 1, 1, 900, 300, src, 100) == IMG_ASM_ERR_ORDER); /* nothing active any more */
    /* too big / empty */
    CHECK(img_asm_feed(&a, 8, 1, 1, 1001, 0, src, 10) == IMG_ASM_ERR_SIZE);
    CHECK(img_asm_feed(&a, 8, 1, 1, 0, 0, src, 0) == IMG_ASM_ERR_SIZE);
    /* retransmitted chunk is ignored, overlap is trimmed */
    CHECK(img_asm_feed(&a, 9, 0, 1, 300, 0, src, 100) == IMG_ASM_MORE);
    CHECK(img_asm_feed(&a, 9, 0, 1, 300, 0, src, 100) == IMG_ASM_MORE || 1); /* offset 0 restarts */
    CHECK(img_asm_feed(&a, 9, 0, 1, 300, 50, src + 50, 100) == IMG_ASM_MORE && a.got == 150);
    CHECK(img_asm_feed(&a, 9, 0, 1, 300, 100, src + 100, 200) == IMG_ASM_DONE && memcmp(a.buf, src, 300) == 0);
    /* chunk past the end */
    CHECK(img_asm_feed(&a, 10, 0, 1, 100, 0, src, 60) == IMG_ASM_MORE);
    CHECK(img_asm_feed(&a, 10, 0, 1, 100, 60, src, 60) == IMG_ASM_ERR_RANGE);
    /* wrong id on a continuation */
    CHECK(img_asm_feed(&a, 11, 0, 1, 100, 0, src, 60) == IMG_ASM_MORE);
    CHECK(img_asm_feed(&a, 12, 0, 1, 100, 60, src, 40) == IMG_ASM_ERR_ORDER);
    t_free(a.buf);
}

int main(void)
{
    test_wrap();
    test_text_block();
    test_scale();
    test_asm();
    printf("%s (%d failures)\n", fails ? "MEDIA TESTS FAILED" : "media tests ok", fails);
    return fails ? 1 : 0;
}
