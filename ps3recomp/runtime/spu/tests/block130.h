/* ---- v130: float compares / conversions / estimates and quadword BIT shifts as whole-register ops ----
 * Profiling inFamous (2026-10-10): the main thread waits 57-71% of each frame on CTaskMgr/WWS SPU job chains, and
 * the lanes running them spend 73-90% in lifted code. The hottest SIMD kernels (e.g. jobmod_2e6aac472eee
 * func_F8C0: ~140 ops/iteration) still went through SCALAR per-lane helpers for fcgt / cflts / shlqbii / fi ...,
 * which on the rc sources' u128 locals become scalar insert/extract chains (see the header comment), and spu_fi
 * called libm ldexpf() twice per lane. Every function below is bit-identical to its spu_helpers.h reference under
 * every MXCSR mode (round mode, FTZ, DAZ) and both g_spu_xfloat settings -- tests/test_spu_helpers_v130.c checks
 * them against the references on edge-case sweeps and random inputs. Where a lane would need a corner case that
 * is awkward to vectorise exactly (NaN into a conversion, fi with a tiny exponent), the reference is called for
 * that register instead (same result by definition). Requires AVX2 + FMA (the lifted SPU TUs are built with
 * -mavx2 -mfma); otherwise the references stay in use. */
#if defined(__AVX2__) && defined(__FMA__) && defined(SPU__FSIMD) && SPU__FSIMD && !defined(SPU_HELPERS_V130_DISABLE)
#include <immintrin.h>
static inline u128 spu__v_fcgt(u128 a, u128 b)  { return spu__fst(_mm_cmpgt_ps(spu__xfq(spu__fld(a)), spu__xfq(spu__fld(b)))); }
static inline u128 spu__v_fceq(u128 a, u128 b)  { return spu__fst(_mm_cmpeq_ps(spu__xfq(spu__fld(a)), spu__xfq(spu__fld(b)))); }
static inline __m128 spu__v_fabs(__m128 v) { return _mm_and_ps(v, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF))); }
static inline u128 spu__v_fcmgt(u128 a, u128 b) { return spu__fst(_mm_cmpgt_ps(spu__v_fabs(spu__xfq(spu__fld(a))), spu__v_fabs(spu__xfq(spu__fld(b))))); }
static inline u128 spu__v_fcmeq(u128 a, u128 b) { return spu__fst(_mm_cmpeq_ps(spu__v_fabs(spu__xfq(spu__fld(a))), spu__v_fabs(spu__xfq(spu__fld(b))))); }
/* cflts: (int32) clamp((double)x * 2^(173-i8)); NaN lanes -> reference (its (int32_t)NaN is compiler-defined). */
static inline u128 spu__v_cflts(u128 a, int i8) {
    const __m128 A = spu__xfq(spu__fld(a));
    if (_mm_movemask_ps(_mm_cmpunord_ps(A, A))) return spu_cflts(a, i8);
    __m256d d = _mm256_mul_pd(_mm256_cvtps_pd(A), _mm256_set1_pd(spu__p2(173 - i8)));
    d = _mm256_max_pd(_mm256_min_pd(d, _mm256_set1_pd(2147483647.0)), _mm256_set1_pd(-2147483648.0));
    return spu__vo(_mm256_cvttpd_epi32(d));
}
/* csflt / cuflt: (float)((double)int * 2^(i8-155)) -- exact product, one double->float rounding (MXCSR), as the reference. */
static inline u128 spu__v_csflt(u128 a, int i8) {
    const __m256d d = _mm256_mul_pd(_mm256_cvtepi32_pd(spu__vi(a)), _mm256_set1_pd(spu__p2(i8 - 155)));
    return spu__fst(_mm256_cvtpd_ps(d));
}
static inline u128 spu__v_cuflt(u128 a, int i8) {
    __m256d d = _mm256_cvtepi32_pd(spu__vi(a));
    d = _mm256_add_pd(d, _mm256_and_pd(_mm256_cmp_pd(d, _mm256_setzero_pd(), _CMP_LT_OQ), _mm256_set1_pd(4294967296.0)));
    return spu__fst(_mm256_cvtpd_ps(_mm256_mul_pd(d, _mm256_set1_pd(spu__p2(i8 - 155)))));
}
/* fi: base - step * y, with step = (rb & 0x3FF) * 2^(exp-140), y = (ra & 0x7FFFF) * 2^-19 (both EXACT for
 * exp >= 14: no rounding, no subnormal), sign of step = sign of base (base < 0.0f). The reference expression
 * `base - step * y` is contracted to one fused op by clang (-mfma, default -ffp-contract=on) -> vfnmadd here.
 * Any lane with exp < 14 (step could be subnormal: ldexpf/DAZ corner cases) -> reference for the register. */
static inline u128 spu__v_fi(u128 a, u128 b) {
    const __m128i A = spu__vi(a), B = spu__vi(b);
    const __m128i e = _mm_and_si128(_mm_srli_epi32(B, 23), _mm_set1_epi32(0xFF));
    if (_mm_movemask_ps(_mm_castsi128_ps(_mm_cmplt_epi32(e, _mm_set1_epi32(14))))) return spu_fi(a, b);
    const __m128 base = _mm_castsi128_ps(_mm_and_si128(B, _mm_set1_epi32((int)0xFFFFFC00u)));
    const __m128 p2 = _mm_castsi128_ps(_mm_slli_epi32(_mm_sub_epi32(e, _mm_set1_epi32(13)), 23));
    __m128 step = _mm_mul_ps(_mm_cvtepi32_ps(_mm_and_si128(B, _mm_set1_epi32(0x3FF))), p2);
    step = _mm_xor_ps(step, _mm_and_ps(_mm_cmplt_ps(base, _mm_setzero_ps()), _mm_castsi128_ps(_mm_set1_epi32((int)0x80000000u))));
    const __m128 y = _mm_mul_ps(_mm_cvtepi32_ps(_mm_and_si128(A, _mm_set1_epi32(0x7FFFF))), _mm_set1_ps(1.0f / 524288.0f));
    return spu__fst(_mm_fnmadd_ps(step, y, base));
}
/* frest / frsqest: the same exponent rules and fraction LUTs as the references, lanes in parallel (gather). */
static inline u128 spu__v_frest(u128 a) {
    const __m128i A = spu__vi(a);
    const __m128i sign = _mm_and_si128(A, _mm_set1_epi32((int)0x80000000u));
    const __m128i ex = _mm_and_si128(_mm_srli_epi32(A, 23), _mm_set1_epi32(0xFF));
    const __m128i fi = _mm_and_si128(_mm_srli_epi32(A, 18), _mm_set1_epi32(0x1F));
    const __m128i fr = _mm_and_si128(_mm_i32gather_epi32((const int*)spu_frest_fraction_lut, fi, 4), _mm_set1_epi32(0x7FFFFF));
    const __m128i big = _mm_cmpgt_epi32(ex, _mm_set1_epi32(252));                       /* exp >= 253 -> rexp 0 */
    const __m128i rexp = _mm_andnot_si128(big, _mm_sub_epi32(_mm_set1_epi32(253), ex));
    const __m128i norm = _mm_or_si128(sign, _mm_or_si128(_mm_slli_epi32(rexp, 23), fr));
    const __m128i zero = _mm_cmpeq_epi32(ex, _mm_setzero_si128());                      /* exp == 0 -> sign|0x7FFFFFFF */
    return spu__vo(_mm_blendv_epi8(norm, _mm_or_si128(sign, _mm_set1_epi32(0x7FFFFFFF)), zero));
}
static inline u128 spu__v_frsqest(u128 a) {
    const __m128i A = spu__vi(a);
    const __m128i ex = _mm_and_si128(_mm_srli_epi32(A, 23), _mm_set1_epi32(0xFF));
    const __m128i fi = _mm_and_si128(_mm_srli_epi32(A, 18), _mm_set1_epi32(0x3F));
    const __m128i fr = _mm_and_si128(_mm_i32gather_epi32((const int*)spu_frsqest_fraction_lut, fi, 4), _mm_set1_epi32(0x7FFFFF));
    __m128i rexp = _mm_sub_epi32(_mm_set1_epi32(190), _mm_srli_epi32(_mm_add_epi32(ex, _mm_set1_epi32(1)), 1));
    rexp = _mm_blendv_epi8(rexp, _mm_set1_epi32(0xFF), _mm_cmpeq_epi32(ex, _mm_setzero_si128()));
    return spu__vo(_mm_or_si128(_mm_slli_epi32(rexp, 23), fr));
}
/* Quadword BIT shifts/rotates by 0..7 (word 0 = most significant, as the references assemble it). */
static inline u128 spu__v_qshl_bits(u128 a, int sh, int rot) {
    const __m128i A = spu__vi(a);
    const __m128i N = rot ? _mm_shuffle_epi32(A, _MM_SHUFFLE(0, 3, 2, 1)) : _mm_srli_si128(A, 4);   /* lane i <- lane i+1 */
    const __m128i c1 = _mm_cvtsi32_si128(sh), c2 = _mm_cvtsi32_si128(32 - sh);
    return spu__vo(_mm_or_si128(_mm_sll_epi32(A, c1), _mm_srl_epi32(N, c2)));
}
static inline u128 spu__v_qshr_bits(u128 a, int sh) {
    const __m128i A = spu__vi(a);
    const __m128i P = _mm_slli_si128(A, 4);                                                          /* lane i <- lane i-1 */
    const __m128i c1 = _mm_cvtsi32_si128(sh), c2 = _mm_cvtsi32_si128(32 - sh);
    return spu__vo(_mm_or_si128(_mm_srl_epi32(A, c1), _mm_sll_epi32(P, c2)));
}
static inline u128 spu__v_shlqbii(u128 a, int sh)  { return spu__v_qshl_bits(a, sh & 7, 0); }
static inline u128 spu__v_rotqbii(u128 a, int sh)  { return spu__v_qshl_bits(a, sh & 7, 1); }
static inline u128 spu__v_shlqbi(u128 a, u128 b)   { return spu__v_qshl_bits(a, (int)(b._u32[0] & 7), 0); }
static inline u128 spu__v_rotqbi(u128 a, u128 b)   { return spu__v_qshl_bits(a, (int)(b._u32[0] & 7), 1); }
static inline u128 spu__v_rotqmbii(u128 a, int i7) { return spu__v_qshr_bits(a, (0 - i7) & 7); }
static inline u128 spu__v_rotqmbi(u128 a, u128 b)  { return spu__v_qshr_bits(a, (0 - (int)b._u32[0]) & 7); }
#ifndef SPU_HELPERS_V130_NO_REDIRECT
#define spu_fcgt    spu__v_fcgt
#define spu_fceq    spu__v_fceq
#define spu_fcmgt   spu__v_fcmgt
#define spu_fcmeq   spu__v_fcmeq
#define spu_cflts   spu__v_cflts
#define spu_csflt   spu__v_csflt
#define spu_cuflt   spu__v_cuflt
#define spu_fi      spu__v_fi
/* frest: gather is slower than the reference on current CPUs -- not redirected */
/* frsqest: idem */
#define spu_shlqbii spu__v_shlqbii
#define spu_rotqbii spu__v_rotqbii
#define spu_shlqbi  spu__v_shlqbi
#define spu_rotqbi  spu__v_rotqbi
#define spu_rotqmbii spu__v_rotqmbii
#define spu_rotqmbi spu__v_rotqmbi
#endif
#endif /* v130 */
