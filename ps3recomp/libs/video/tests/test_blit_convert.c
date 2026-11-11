#include "../rsx_blit_convert.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng_state = 0x13579BDFu;
static uint32_t next_u32(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

int main(void)
{
    /* Odd widths exercise AVX2, SSE2 and scalar tails. Unequal padded pitches
     * model D3D's 256-byte readback rows and the guest's packed destination. */
    static const uint32_t widths[] = {1, 3, 4, 7, 8, 9, 255, 256, 1024, 1280};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        const uint32_t w = widths[wi], h = 7;
        const size_t sp = ((size_t)w * 4u + 255u) & ~(size_t)255u;
        const size_t dp = (size_t)w * 4u + 20u;
        uint8_t* src = (uint8_t*)malloc(sp * h);
        uint8_t* got = (uint8_t*)malloc(dp * h);
        uint8_t* ref = (uint8_t*)malloc(dp * h);
        if (!src || !got || !ref) return 2;
        for (size_t i = 0; i < sp * h; i++) src[i] = (uint8_t)next_u32();
        memset(got, 0xCD, dp * h); memset(ref, 0xCD, dp * h);
        rsx_blit_rgba_to_argb_rows(src, sp, got, dp, w, h);
        for (uint32_t y = 0; y < h; y++) {
            const uint32_t* q = (const uint32_t*)(src + (size_t)y * sp);
            uint32_t* o = (uint32_t*)(ref + (size_t)y * dp);
            for (uint32_t x = 0; x < w; x++) o[x] = (q[x] << 8) | (q[x] >> 24);
        }
        if (memcmp(got, ref, dp * h) != 0) {
            fprintf(stderr, "mismatch at width %u\n", w);
            return 1;
        }
        free(src); free(got); free(ref);
    }
    puts("blit conversion: PASS");
    return 0;
}
