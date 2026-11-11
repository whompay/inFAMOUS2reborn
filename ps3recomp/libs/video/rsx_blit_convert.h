#ifndef PS3RECOMP_RSX_BLIT_CONVERT_H
#define PS3RECOMP_RSX_BLIT_CONVERT_H

#include <stddef.h>
#include <stdint.h>

#if defined(__AVX2__)
#include <immintrin.h>
#define RSX_BLIT_CONVERT_AVX2 1
#endif
#if defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define RSX_BLIT_CONVERT_SSE2 1
#endif

/* D3D readback bytes are R,G,B,A. Guest A8R8G8B8 memory expects A,R,G,B,
 * which is a rotate-left-by-eight of each little-endian 32-bit pixel. Pitches
 * are bytes and may include independent source/destination row padding. */
static inline void rsx_blit_rgba_to_argb_rows(const void* src_data,
                                               size_t src_pitch,
                                               void* dst_data,
                                               size_t dst_pitch,
                                               uint32_t width,
                                               uint32_t height)
{
    const uint8_t* src = (const uint8_t*)src_data;
    uint8_t* dst = (uint8_t*)dst_data;
    for (uint32_t y = 0; y < height; y++) {
        const uint32_t* q = (const uint32_t*)(src + (size_t)y * src_pitch);
        uint32_t* o = (uint32_t*)(dst + (size_t)y * dst_pitch);
        uint32_t x = 0;
#if defined(RSX_BLIT_CONVERT_AVX2)
        for (; x + 8u <= width; x += 8u) {
            const __m256i v = _mm256_loadu_si256((const __m256i*)(q + x));
            const __m256i r = _mm256_or_si256(_mm256_slli_epi32(v, 8),
                                              _mm256_srli_epi32(v, 24));
            _mm256_storeu_si256((__m256i*)(o + x), r);
        }
#endif
#if defined(RSX_BLIT_CONVERT_SSE2)
        for (; x + 4u <= width; x += 4u) {
            const __m128i v = _mm_loadu_si128((const __m128i*)(q + x));
            const __m128i r = _mm_or_si128(_mm_slli_epi32(v, 8),
                                           _mm_srli_epi32(v, 24));
            _mm_storeu_si128((__m128i*)(o + x), r);
        }
#endif
        for (; x < width; x++) {
            const uint32_t v = q[x];
            o[x] = (v << 8) | (v >> 24);
        }
    }
}

#endif
