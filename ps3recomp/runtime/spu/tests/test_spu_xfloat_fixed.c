/* Bit-exact check for the opt-in fixed-on clamp used by generated RC images.
 * Build with clang-cl -O2 -msse4.1 -mavx2 -DSPU_RC_XFLOAT_FIXED -I../ . */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int g_spu_xfloat = 1;
uint32_t g_spu_xflim[8] = {
    0xFF7FFFFFu, 0xFF7FFFFFu, 0xFF7FFFFFu, 0xFF7FFFFFu,
    0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu
};
#include "spu_helpers.h"

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t next32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 9);
}
static __m128 dynamic_clamp(__m128 v) {
    __m128i x = _mm_castps_si128(v);
    x = _mm_min_epu32(x, _mm_loadu_si128((const __m128i*)&g_spu_xflim[0]));
    x = _mm_min_epi32(x, _mm_loadu_si128((const __m128i*)&g_spu_xflim[4]));
    return _mm_castsi128_ps(x);
}
static int check(const uint32_t a[4], unsigned csr) {
    __m128 v; memcpy(&v, a, 16);
    __m128 x = spu__xfq(v), y = dynamic_clamp(v);
    uint32_t xb[4], yb[4]; memcpy(xb, &x, 16); memcpy(yb, &y, 16);
    if (!memcmp(xb, yb, 16)) return 0;
    fprintf(stderr, "xfloat mismatch csr=%04X input=%08X fixed=%08X dynamic=%08X\n",
            csr, a[0], xb[0], yb[0]);
    return 1;
}
int main(void) {
    const unsigned saved = _mm_getcsr();
    uint64_t count = 0;
    for (unsigned rm = 0; rm < 4; ++rm) for (unsigned fz = 0; fz < 4; ++fz) {
        const unsigned csr = (saved & 0x3Fu) | 0x1F80u | (rm << 13)
            | ((fz & 1) ? 0x8000u : 0) | ((fz & 2) ? 0x40u : 0);
        _mm_setcsr(csr);
        /* Every sign/exponent combination, with the mantissas at exact
         * clamp boundaries and on both sides of them. */
        for (uint32_t e = 0; e < 256; ++e) for (uint32_t sign = 0; sign < 2; ++sign) {
            const uint32_t a[4] = {
                (sign << 31) | (e << 23),
                (sign << 31) | (e << 23) | 1u,
                (sign << 31) | (e << 23) | 0x7FFFFEu,
                (sign << 31) | (e << 23) | 0x7FFFFFu
            };
            if (check(a, csr)) return 1;
            ++count;
        }
        for (uint32_t i = 0; i < 1000000; ++i) {
            uint32_t a[4] = { next32(), next32(), next32(), next32() };
            if (check(a, csr)) return 1;
            ++count;
        }
    }
    _mm_setcsr(saved);
    printf("xfloat fixed-on clamp: %llu vectors (%llu lanes), 0 mismatches\n",
           (unsigned long long)count, (unsigned long long)(4 * count));
    return 0;
}
