#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
int g_spu_xfloat = 1;
uint32_t g_spu_xflim[8] = { 0xFF7FFFFFu, 0xFF7FFFFFu, 0xFF7FFFFFu, 0xFF7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu };
#include "spu_helpers.h"
#define SPU_HELPERS_V130_NO_REDIRECT
#include <immintrin.h>
static inline __m128i spu__vi(u128 a) { __m128i x; memcpy(&x, &a, 16); return x; }
static inline u128 spu__vo(__m128i x) { u128 a; memcpy(&a, &x, 16); return a; }
/* block130.h = the v130 section of spu_helpers_vec.h (build: clang -O2 -msse4.1 -mavx2 -mfma -mbmi2 -I.. -I../../../include) */
#include "block130.h"
static uint64_t s = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void){ s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 11); }
static uint32_t special(void){
    static const uint32_t sp[] = {0,0x80000000u,1,0x80000001u,0x007FFFFFu,0x807FFFFFu,0x00800000u,0x7F7FFFFFu,0xFF7FFFFFu,0x7F800000u,0xFF800000u,
        0x7FC00000u,0xFFC00000u,0x7F800001u,0x3F800000u,0xBF800000u,0x4F000000u,0xCF000000u,0x4EFFFFFFu,0x4F800000u,0x5F000000u,0x2F800000u,0x00000400u,0x06FFFFFFu,0x07000000u,0x07FFFFFFu};
    uint32_t r = rnd32();
    switch (r & 7) { case 0: case 1: return sp[(r >> 3) % (sizeof sp / 4)];
      case 2: return (r & 0x807FFFFFu) | ((uint32_t)((r >> 9) % 3 ? (r >> 12) & 0xFF : ((r>>12)&1 ? 0xFF : 0)) << 23);
      case 3: return (r & 0x80000000u) | (((r >> 4) & 0xFF) << 23) | (r & 0x7FFFF);
      default: return rnd32(); }
}
static u128 rv(void){ u128 r; for(int i=0;i<4;i++) r._u32[i] = special(); return r; }
static long bad = 0, tot = 0;
static void chk(const char* n, u128 x, u128 y, u128 a, u128 b, int k, unsigned csr){
    tot++;
    if (!strcmp(n,"cuflt")) for(int i=0;i<4;i++) if (a._u32[i]==0 && x._u32[i]==0x80000000u) x._u32[i]=0; /* clang magic-number u32->f64 gives -0 in RD; +0 is SPU-correct */
    if (memcmp(&x, &y, 16)) { if (bad++ < 40) printf("MISMATCH %s k=%d csr=%04X a=%08X %08X %08X %08X b=%08X.. ref=%08X %08X %08X %08X vec=%08X %08X %08X %08X\n",
        n, k, csr, a._u32[0],a._u32[1],a._u32[2],a._u32[3], b._u32[0], x._u32[0],x._u32[1],x._u32[2],x._u32[3], y._u32[0],y._u32[1],y._u32[2],y._u32[3]); }
}
int main(int argc, char** argv){
    const long N = argc > 1 ? atol(argv[1]) : 20000;
    const unsigned base_csr = _mm_getcsr() & ~0xFFC0u;
    for (int xf = 0; xf < 2; xf++) { g_spu_xfloat = xf; for(int q=0;q<4;q++){ g_spu_xflim[q]= xf?0xFF7FFFFFu:0xFFFFFFFFu; g_spu_xflim[4+q]= xf?0x7F7FFFFFu:0x7FFFFFFFu; }
      for (unsigned rm = 0; rm < 4; rm++) for (unsigned fz = 0; fz < 4; fz++) {
        const unsigned csr = (base_csr & 0x3Fu) | 0x1F80u | (rm << 13) | ((fz & 1) ? 0x8000u : 0) | ((fz & 2) ? 0x40u : 0);
        _mm_setcsr(csr);
        for (long it = 0; it < N; it++) {
            u128 a = rv(), b = rv(); int i8 = (int)(rnd32() & 0xFF), sh = (int)(rnd32() & 0x7F);
            chk("fcgt", spu_fcgt(a,b), spu__v_fcgt(a,b), a,b,0,csr);
            chk("fceq", spu_fceq(a,b), spu__v_fceq(a,b), a,b,0,csr);
            chk("fcmgt", spu_fcmgt(a,b), spu__v_fcmgt(a,b), a,b,0,csr);
            chk("fcmeq", spu_fcmeq(a,b), spu__v_fcmeq(a,b), a,b,0,csr);
            u128 bb = b; if (it & 1) for(int i=0;i<4;i++) bb._u32[i] = a._u32[i];   /* fceq/fcmeq equal cases */
            chk("fceq=", spu_fceq(a,bb), spu__v_fceq(a,bb), a,bb,0,csr);
            chk("fcmeq=", spu_fcmeq(a,bb), spu__v_fcmeq(a,bb), a,bb,0,csr);
            chk("cflts", spu_cflts(a,i8), spu__v_cflts(a,i8), a,b,i8,csr);
            chk("cflts173", spu_cflts(a,173), spu__v_cflts(a,173), a,b,173,csr);
            chk("csflt", spu_csflt(a,i8), spu__v_csflt(a,i8), a,b,i8,csr);
            chk("cuflt", spu_cuflt(a,i8), spu__v_cuflt(a,i8), a,b,i8,csr);
            chk("csflt155", spu_csflt(a,155), spu__v_csflt(a,155), a,b,155,csr);
            chk("fi", spu_fi(a,b), spu__v_fi(a,b), a,b,0,csr);
            { u128 b2 = b; for(int i=0;i<4;i++) b2._u32[i] = (b2._u32[i] & 0x807FFFFFu) | ((14u + (rnd32() % 242u)) << 23);   /* force the vector path */
              chk("fi-vec", spu_fi(a,b2), spu__v_fi(a,b2), a,b2,0,csr); }
            chk("frest", spu_frest(a), spu__v_frest(a), a,b,0,csr);
            chk("frsqest", spu_frsqest(a), spu__v_frsqest(a), a,b,0,csr);
            chk("shlqbii", spu_shlqbii(a,sh), spu__v_shlqbii(a,sh), a,b,sh,csr);
            chk("rotqbii", spu_rotqbii(a,sh), spu__v_rotqbii(a,sh), a,b,sh,csr);
            chk("rotqmbii", spu_rotqmbii(a,-sh), spu__v_rotqmbii(a,-sh), a,b,-sh,csr);
            chk("rotqmbii+", spu_rotqmbii(a,sh), spu__v_rotqmbii(a,sh), a,b,sh,csr);
            chk("shlqbi", spu_shlqbi(a,b), spu__v_shlqbi(a,b), a,b,0,csr);
            chk("rotqbi", spu_rotqbi(a,b), spu__v_rotqbi(a,b), a,b,0,csr);
            chk("rotqmbi", spu_rotqmbi(a,b), spu__v_rotqmbi(a,b), a,b,0,csr);
        }
        /* exhaustive: every exponent x every i8 for the conversions; every LUT index for the estimates */
        for (uint32_t e = 0; e < 256; e++) for (int k = 0; k < 256; k += 1) {
            u128 a; for(int i=0;i<4;i++) a._u32[i] = (e << 23) | (rnd32() & 0x807FFFFFu);
            chk("cflts-x", spu_cflts(a,k), spu__v_cflts(a,k), a,a,k,csr);
            u128 ia; for(int i=0;i<4;i++) ia._u32[i] = rnd32() >> (rnd32() & 31);
            chk("csflt-x", spu_csflt(ia,k), spu__v_csflt(ia,k), ia,ia,k,csr);
            chk("cuflt-x", spu_cuflt(ia,k), spu__v_cuflt(ia,k), ia,ia,k,csr);
        }
        for (uint32_t w = 0; w < (1u << 14); w++) { u128 a; for(int i=0;i<4;i++) a._u32[i] = (w << 18) | (rnd32() & 0x3FFFFu) | (i == 3 ? 0x80000000u : 0);
            chk("frest-x", spu_frest(a), spu__v_frest(a), a,a,0,csr); chk("frsqest-x", spu_frsqest(a), spu__v_frsqest(a), a,a,0,csr); }
      } }
    printf("%ld checks, %ld mismatches\n", tot, bad);
    return bad != 0;
}
