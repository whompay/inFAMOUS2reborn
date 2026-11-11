/*
 * ps3recomp - SPU instruction semantics helpers
 *
 * Pure-C, header-only implementation of the per-instruction semantics used
 * by lifter-generated code. Extracted from spu_lifter.py so the helpers
 * have one source of truth and can be unit-tested directly (see
 * runtime/spu/tests/test_spu_helpers.c).
 *
 * Each helper is `static inline u128 spu_<mnemonic>(...)`. Naming, lane
 * widths and big-endian conventions match runtime/spu/spu_context.h.
 */

#ifndef SPU_HELPERS_H
#define SPU_HELPERS_H

#include "spu_context.h"
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#include <intrin.h>
static inline int spu_clz32(uint32_t x) {
    unsigned long idx;
    if (!x) return 32;
    _BitScanReverse(&idx, x);
    return 31 - (int)idx;
}
#else
static inline int spu_clz32(uint32_t x) { return x ? __builtin_clz(x) : 32; }
#endif

/* Architectural SPU barrier -- the lifted `sync`/`dsync`/`syncc` emission.
 * SPU ISA v1.2 §13 (+ the per-instruction pages): order all earlier LS and
 * channel accesses before later ones; LS must be consistent "if it were\n * observed by another entity" (Table 13-6 is exactly this runtime's
 * host-thread producer -> SPU consumer handoff). A no-op is neither a
 * compiler nor a CPU barrier -- MSVC at /O2 may reorder plain LS accesses
 * across an empty statement (the PPU twin of this bug is LESSONS #11d). A
 * full fence: the locked exchange forces total order and is opaque to the
 * optimizer. */
#ifdef _MSC_VER
static inline void spu_arch_fence(void) { long spu_fence_tmp = 0; _InterlockedExchange(&spu_fence_tmp, 1); }
#else
static inline void spu_arch_fence(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
#endif

/* SPU byte position -> our _u8 index.
 * Our u128 is HOST-NATIVE little-endian (_u32[i]=SPU word i as a value via
 * spu_ls_read128's big-endian load), so within each 4-byte word the _u8[]
 * bytes are reversed vs SPU big-endian byte order. Any op defined in terms of
 * SPU *byte positions* (quadword byte rotates/shifts, gen-controls, shuffle
 * insertion) must map SPU byte P to our index W(P). Word-aligned operations
 * are unaffected (W is identity modulo the within-word reversal). */
#define SPU_W(P) (((P) & ~3) | (3 - ((P) & 3)))


/* v30b (FPS): byte-permute helpers as ONE pshufb with a precomputed control.
 * The scalar versions (kept as *_ref, used when SSSE3 is unavailable and by the
 * equivalence test _xfer/test_helpers_simd.c) moved 16 bytes one at a time
 * through the SPU_W index map -- rotqby/rotqbyi/shlqbyi alone are ~1/4 of the
 * lifted job modules' quadword ops (every unaligned scalar load), and cwd/cbd
 * (every scalar store) rebuilt a 16-byte control in a loop. Bit-exact vs the
 * reference for every shift / position (exhaustive test). */
#if defined(__SSSE3__) || defined(__SSE4_1__) || defined(__AVX__)
#include <tmmintrin.h>
#define SPU__SIMD 1
#if defined(_MSC_VER) && !defined(__clang__)
#define SPU__ALIGN16 __declspec(align(16))
#else
#define SPU__ALIGN16 __attribute__((aligned(16)))
#endif
static const SPU__ALIGN16 uint8_t spu__t_rot[16][16] = {
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F},
    {0x07,0x00,0x01,0x02,0x0B,0x04,0x05,0x06,0x0F,0x08,0x09,0x0A,0x03,0x0C,0x0D,0x0E},
    {0x06,0x07,0x00,0x01,0x0A,0x0B,0x04,0x05,0x0E,0x0F,0x08,0x09,0x02,0x03,0x0C,0x0D},
    {0x05,0x06,0x07,0x00,0x09,0x0A,0x0B,0x04,0x0D,0x0E,0x0F,0x08,0x01,0x02,0x03,0x0C},
    {0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0x00,0x01,0x02,0x03},
    {0x0B,0x04,0x05,0x06,0x0F,0x08,0x09,0x0A,0x03,0x0C,0x0D,0x0E,0x07,0x00,0x01,0x02},
    {0x0A,0x0B,0x04,0x05,0x0E,0x0F,0x08,0x09,0x02,0x03,0x0C,0x0D,0x06,0x07,0x00,0x01},
    {0x09,0x0A,0x0B,0x04,0x0D,0x0E,0x0F,0x08,0x01,0x02,0x03,0x0C,0x05,0x06,0x07,0x00},
    {0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07},
    {0x0F,0x08,0x09,0x0A,0x03,0x0C,0x0D,0x0E,0x07,0x00,0x01,0x02,0x0B,0x04,0x05,0x06},
    {0x0E,0x0F,0x08,0x09,0x02,0x03,0x0C,0x0D,0x06,0x07,0x00,0x01,0x0A,0x0B,0x04,0x05},
    {0x0D,0x0E,0x0F,0x08,0x01,0x02,0x03,0x0C,0x05,0x06,0x07,0x00,0x09,0x0A,0x0B,0x04},
    {0x0C,0x0D,0x0E,0x0F,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B},
    {0x03,0x0C,0x0D,0x0E,0x07,0x00,0x01,0x02,0x0B,0x04,0x05,0x06,0x0F,0x08,0x09,0x0A},
    {0x02,0x03,0x0C,0x0D,0x06,0x07,0x00,0x01,0x0A,0x0B,0x04,0x05,0x0E,0x0F,0x08,0x09},
    {0x01,0x02,0x03,0x0C,0x05,0x06,0x07,0x00,0x09,0x0A,0x0B,0x04,0x0D,0x0E,0x0F,0x08},
};
static const SPU__ALIGN16 uint8_t spu__t_shl[16][16] = {
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F},
    {0x07,0x00,0x01,0x02,0x0B,0x04,0x05,0x06,0x0F,0x08,0x09,0x0A,0x80,0x0C,0x0D,0x0E},
    {0x06,0x07,0x00,0x01,0x0A,0x0B,0x04,0x05,0x0E,0x0F,0x08,0x09,0x80,0x80,0x0C,0x0D},
    {0x05,0x06,0x07,0x00,0x09,0x0A,0x0B,0x04,0x0D,0x0E,0x0F,0x08,0x80,0x80,0x80,0x0C},
    {0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0x80,0x80,0x80,0x80},
    {0x0B,0x04,0x05,0x06,0x0F,0x08,0x09,0x0A,0x80,0x0C,0x0D,0x0E,0x80,0x80,0x80,0x80},
    {0x0A,0x0B,0x04,0x05,0x0E,0x0F,0x08,0x09,0x80,0x80,0x0C,0x0D,0x80,0x80,0x80,0x80},
    {0x09,0x0A,0x0B,0x04,0x0D,0x0E,0x0F,0x08,0x80,0x80,0x80,0x0C,0x80,0x80,0x80,0x80},
    {0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x0F,0x08,0x09,0x0A,0x80,0x0C,0x0D,0x0E,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x0E,0x0F,0x08,0x09,0x80,0x80,0x0C,0x0D,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x0D,0x0E,0x0F,0x08,0x80,0x80,0x80,0x0C,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x0C,0x0D,0x0E,0x0F,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x80,0x0C,0x0D,0x0E,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x80,0x80,0x0C,0x0D,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x80,0x80,0x80,0x0C,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
};
static const SPU__ALIGN16 uint8_t spu__t_shr[16][16] = {
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F},
    {0x01,0x02,0x03,0x80,0x05,0x06,0x07,0x00,0x09,0x0A,0x0B,0x04,0x0D,0x0E,0x0F,0x08},
    {0x02,0x03,0x80,0x80,0x06,0x07,0x00,0x01,0x0A,0x0B,0x04,0x05,0x0E,0x0F,0x08,0x09},
    {0x03,0x80,0x80,0x80,0x07,0x00,0x01,0x02,0x0B,0x04,0x05,0x06,0x0F,0x08,0x09,0x0A},
    {0x80,0x80,0x80,0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B},
    {0x80,0x80,0x80,0x80,0x01,0x02,0x03,0x80,0x05,0x06,0x07,0x00,0x09,0x0A,0x0B,0x04},
    {0x80,0x80,0x80,0x80,0x02,0x03,0x80,0x80,0x06,0x07,0x00,0x01,0x0A,0x0B,0x04,0x05},
    {0x80,0x80,0x80,0x80,0x03,0x80,0x80,0x80,0x07,0x00,0x01,0x02,0x0B,0x04,0x05,0x06},
    {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07},
    {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x01,0x02,0x03,0x80,0x05,0x06,0x07,0x00},
    {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x02,0x03,0x80,0x80,0x06,0x07,0x00,0x01},
    {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x03,0x80,0x80,0x80,0x07,0x00,0x01,0x02},
    {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x00,0x01,0x02,0x03},
    {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x01,0x02,0x03,0x80},
    {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x02,0x03,0x80,0x80},
    {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x03,0x80,0x80,0x80},
};
static const SPU__ALIGN16 uint8_t spu__t_cbd[16][16] = {
    {0x13,0x12,0x11,0x03,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x03,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x03,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x03,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x03,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x03,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x03,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x03,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x03,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x03,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x03},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x03,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x03,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x03,0x1E,0x1D,0x1C},
};
static const SPU__ALIGN16 uint8_t spu__t_chd[16][16] = {
    {0x13,0x12,0x03,0x02,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x03,0x02,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x03,0x02,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x03,0x02,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x03,0x02,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x03,0x02,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x03,0x02,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x03,0x02,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x03,0x02},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x03,0x02},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x03,0x02,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x03,0x02,0x1D,0x1C},
};
static const SPU__ALIGN16 uint8_t spu__t_cwd[16][16] = {
    {0x03,0x02,0x01,0x00,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x01,0x00,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x01,0x00,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x01,0x00,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x03,0x02,0x01,0x00,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x03,0x02,0x01,0x00,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x03,0x02,0x01,0x00,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x03,0x02,0x01,0x00,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x03,0x02,0x01,0x00},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x03,0x02,0x01,0x00},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x03,0x02,0x01,0x00},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x1B,0x1A,0x19,0x18,0x03,0x02,0x01,0x00},
};
static const SPU__ALIGN16 uint8_t spu__t_cdd[16][16] = {
    {0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04,0x1B,0x1A,0x19,0x18,0x1F,0x1E,0x1D,0x1C},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04},
    {0x13,0x12,0x11,0x10,0x17,0x16,0x15,0x14,0x03,0x02,0x01,0x00,0x07,0x06,0x05,0x04},
};
static inline u128 spu__pshufb(u128 a, const uint8_t* ctl) {
    __m128i A; memcpy(&A, &a, 16);
    __m128i r = _mm_shuffle_epi8(A, _mm_load_si128((const __m128i*)ctl));
    u128 o; memcpy(&o, &r, 16); return o;
}
static inline u128 spu__ldtbl(const uint8_t* t) { u128 o; memcpy(&o, t, 16); return o; }
#else
#define SPU__SIMD 0
#endif

/* ---- constructors ---- */
static inline u128 spu_splat_u32(uint32_t v) {
    u128 r; r._u32[0]=v; r._u32[1]=v; r._u32[2]=v; r._u32[3]=v; return r;
}
/* Link register value for brsl/bisl/bisled: the SPU writes the return address
 * into the PREFERRED word (word 0) and ZEROS the other three slots (matches
 * RPCS3 v128::from32r, SPUInterpreter.cpp BRSL). Splatting it (the old bug)
 * corrupts any code that saves/reloads/operates on the full 128-bit link. */
/* An SPU instruction the lifter could not translate. It emits a bare comment,
 * so an unsupported opcode on a live path silently does NOTHING and the guest
 * computes garbage from there on -- which surfaces much later as a wild DMA or
 * atomic address with no hint of where it came from. Most of these are data
 * embedded in .text that never executes; this fires only if one actually runs.
 * One line per site. */
static inline void spu_unsupported(uint32_t pc, const char* mnemonic)
{
    static int warned;
    if (warned < 64) {
        warned++;
        fprintf(stderr, "[spu] UNSUPPORTED %s at LS 0x%05X -- executed, "
                        "result is wrong from here\n", mnemonic, pc);
        fflush(stderr);
    }
}

static inline u128 spu_link(uint32_t addr) {
    u128 r; r._u32[0]=addr; r._u32[1]=0; r._u32[2]=0; r._u32[3]=0; return r;
}
/* Preferred-slot scalar: value in word 0, remaining slots ZERO. CBEA's scalar
 * channel convention -- rchcnt returns the count this way (RPCS3 measured
 * {1,0,0,0} where the old splat gave {1,1,1,1}). Same bug class as spu_link. */
static inline u128 spu_pref_u32(uint32_t v) {
    u128 r; r._u32[0]=v; r._u32[1]=0; r._u32[2]=0; r._u32[3]=0; return r;
}
static inline u128 spu_splat_u16(uint16_t v) {
    u128 r; for (int i=0;i<8;i++) r._u16[i]=v; return r;
}
static inline u128 spu_splat_u8(uint8_t v) {
    u128 r; for (int i=0;i<16;i++) r._u8[i]=v; return r;
}
static inline u128 spu_zero(void) { u128 r; memset(&r,0,sizeof r); return r; }

/* ---- integer arithmetic (SIMD) ---- */
static inline u128 spu_a(u128 a, u128 b)  { u128 r; for(int i=0;i<4;i++) r._u32[i]=a._u32[i]+b._u32[i]; return r; }
static inline u128 spu_sf(u128 a, u128 b) { u128 r; for(int i=0;i<4;i++) r._u32[i]=b._u32[i]-a._u32[i]; return r; }
static inline u128 spu_ah(u128 a, u128 b) { u128 r; for(int i=0;i<8;i++) r._u16[i]=a._u16[i]+b._u16[i]; return r; }
static inline u128 spu_sfh(u128 a, u128 b){ u128 r; for(int i=0;i<8;i++) r._u16[i]=b._u16[i]-a._u16[i]; return r; }
static inline u128 spu_ai(u128 a, int32_t imm) { u128 r; for(int i=0;i<4;i++) r._u32[i]=a._u32[i]+(uint32_t)imm; return r; }
static inline u128 spu_ahi(u128 a, int32_t imm){ u128 r; for(int i=0;i<8;i++) r._u16[i]=a._u16[i]+(uint16_t)imm; return r; }
static inline u128 spu_sfi(u128 a, int32_t imm) { u128 r; for(int i=0;i<4;i++) r._u32[i]=(uint32_t)imm-a._u32[i]; return r; }
static inline u128 spu_sfhi(u128 a, int32_t imm){ u128 r; for(int i=0;i<8;i++) r._u16[i]=(uint16_t)imm-a._u16[i]; return r; }

/* ---- multiply (low halfword of each word × ... -> 32-bit, per SPU mpy) ----
 * Sub-lane indexing assumes a little-endian host (matches the recompiler's
 * target). The "low halfword of word i" in SPU BE semantics is _s16[2i] on
 * an LE host (NOT _s16[2i+1], which would be correct on a BE host). */
static inline u128 spu_mpy(u128 a, u128 b)  { u128 r; for(int i=0;i<4;i++) r._s32[i]=(int32_t)a._s16[i*2]*(int32_t)b._s16[i*2]; return r; }
/* mpya: 16x16 signed multiply of low halves + add rc (per word, RRR form). */
static inline u128 spu_mpya(u128 a, u128 b, u128 c) { u128 r; for(int i=0;i<4;i++) r._s32[i]=(int32_t)a._s16[i*2]*(int32_t)b._s16[i*2]+c._s32[i]; return r; }
/* sfx: extended subtract rb-ra-1+carry; carry-in = low bit of old rt (RT is 3rd src). */
static inline u128 spu_sfx(u128 a, u128 b, u128 t) { u128 r; for(int i=0;i<4;i++) r._u32[i]=b._u32[i]+~a._u32[i]+(t._u32[i]&1u); return r; }
/* FI decodes the base and step fields produced by the hardware estimate
 * instructions; its second operand is not an ordinary reciprocal float.
 * NA1000: this is the original libm reference, kept for the SPU_NA_FEAT bit-0 A/B (spu_fi below). */
static inline u128 spu_fi_legacy(u128 a, u128 b) {
    u128 r;
    for (int i = 0; i < 4; i++) {
        uint32_t rb = b._u32[i];
        uint32_t ra = a._u32[i];
        uint32_t exp = (rb >> 23) & 0xFFu;
        uint32_t base_bits = rb & 0xFFFFFC00u;
        float base; memcpy(&base, &base_bits, sizeof base);
        uint32_t step_frac = rb & 0x3FFu;
        float step = ldexpf((float)step_frac, (int)exp - 127 - 13);
        if (base < 0.0f) step = -step;
        float y = ldexpf((float)(ra & 0x7FFFFu), -19);
        r._f32[i] = base - step * y;
    }
    return r;
}
/* NA1000 (perf, 2026-10-16): fi without libm. The native SPU sampler (fps-spu-prof-20261015-224957) put
 * ~20% of all busy SPU-thread samples in fi: the reference above called ucrtbase ldexpf() twice per lane
 * (the lifted job modules include only this header, so every fi there was 8 libm calls), and the rc twins'
 * v130 spu__v_fi fell back to it for a whole register whenever one lane had exp < 14.
 * Bit-identical to spu_fi_legacy under every MXCSR mode:
 *  - y = (ra & 0x7FFFF) * 2^-19 and, for exp >= 14, step = step_frac * 2^(exp-140) are EXACT normal
 *    products (or 0), so rounding mode / FTZ / DAZ cannot change them;
 *  - exp < 14 with a normal step (step_frac >= 2^(14-exp)): two exact scalings, same value;
 *  - exp < 14 with a subnormal step: under DAZ (the SPU run mode, spu_run_with_halt sets RZ+FTZ+DAZ)
 *    the following multiply reads a subnormal step as a signed zero, and the sign is only negated when
 *    base < 0.0f, which under DAZ is false for a zero/denormal base -> step = +0 gives the same bits;
 *    without DAZ libm ldexpf is still called for that lane.
 * SPU_NA_FEAT bit 0 = 0 (or the odd SPU_NA_AB phases) selects spu_fi_legacy again. */
extern int g_spu_na_feat;
#if defined(__GNUC__) || defined(__clang__)
#define SPU__NOINLINE __attribute__((noinline))
#define SPU__AINLINE  inline __attribute__((always_inline))
#define SPU__UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define SPU__NOINLINE __declspec(noinline)
#define SPU__AINLINE  __forceinline
#define SPU__UNLIKELY(x) (x)
#endif
#if defined(__SSE__) || defined(_M_X64) || defined(__x86_64__)
#include <xmmintrin.h>
#define SPU__FI_DAZ() ((_mm_getcsr() & 0x0040u) != 0)
#else
#define SPU__FI_DAZ() 0
#endif
static inline SPU__NOINLINE u128 spu_fi_lanes(u128 a, u128 b) {
    u128 r;
    const int daz = SPU__FI_DAZ();
    for (int i = 0; i < 4; i++) {
        uint32_t rb = b._u32[i];
        uint32_t ra = a._u32[i];
        uint32_t exp = (rb >> 23) & 0xFFu;
        uint32_t base_bits = rb & 0xFFFFFC00u;
        float base; memcpy(&base, &base_bits, sizeof base);
        uint32_t step_frac = rb & 0x3FFu;
        float step;
        if (exp >= 14u) {
            uint32_t pb = (exp - 13u) << 23; float p2; memcpy(&p2, &pb, sizeof p2);       /* 2^(exp-140) */
            step = (float)step_frac * p2;
        } else if (step_frac >= (1u << (14u - exp))) {
            uint32_t pb = (exp + 51u) << 23; float p2; memcpy(&p2, &pb, sizeof p2);       /* 2^(exp-76) */
            step = ((float)step_frac * p2) * 5.42101086242752217e-20f;                     /* * 2^-64 */
        } else if (daz) {
            step = 0.0f;
        } else {
            step = ldexpf((float)step_frac, (int)exp - 127 - 13);
        }
        if (base < 0.0f) step = -step;
        float y = (float)(ra & 0x7FFFFu) * (1.0f / 524288.0f);
        r._f32[i] = base - step * y;
    }
    return r;
}
static inline SPU__NOINLINE u128 spu_fi_slow(u128 a, u128 b) {
    return (g_spu_na_feat & 1) ? spu_fi_lanes(a, b) : spu_fi_legacy(a, b);
}
#if (defined(__AVX2__) && defined(__FMA__)) && (defined(__clang__) || defined(__GNUC__)) && !defined(SPU_FI_VEC_DISABLE)
#include <immintrin.h>
/* whole-register form (same expression as spu_helpers_vec.h v130 spu__v_fi): all lanes exp >= 14 */
static SPU__AINLINE u128 spu_fi(u128 a, u128 b) {
    __m128i A, B; memcpy(&A, &a, 16); memcpy(&B, &b, 16);
    const __m128i e = _mm_and_si128(_mm_srli_epi32(B, 23), _mm_set1_epi32(0xFF));
    if (SPU__UNLIKELY(!(g_spu_na_feat & 1) || _mm_movemask_ps(_mm_castsi128_ps(_mm_cmplt_epi32(e, _mm_set1_epi32(14))))))
        return spu_fi_slow(a, b);
    const __m128 base = _mm_castsi128_ps(_mm_and_si128(B, _mm_set1_epi32((int)0xFFFFFC00u)));
    const __m128 p2 = _mm_castsi128_ps(_mm_slli_epi32(_mm_sub_epi32(e, _mm_set1_epi32(13)), 23));
    __m128 step = _mm_mul_ps(_mm_cvtepi32_ps(_mm_and_si128(B, _mm_set1_epi32(0x3FF))), p2);
    step = _mm_xor_ps(step, _mm_and_ps(_mm_cmplt_ps(base, _mm_setzero_ps()), _mm_castsi128_ps(_mm_set1_epi32((int)0x80000000u))));
    const __m128 y = _mm_mul_ps(_mm_cvtepi32_ps(_mm_and_si128(A, _mm_set1_epi32(0x7FFFF))), _mm_set1_ps(1.0f / 524288.0f));
    /* written as an expression (not _mm_fnmadd_ps) so it contracts exactly like the scalar `base - step * y`
     * under the TU's -ffp-contract setting (fused with the default =on, two roundings with =off) */
    const __m128 res = base - step * y;
    u128 r; memcpy(&r, &res, 16); return r;
}
#else
static inline u128 spu_fi(u128 a, u128 b) { return spu_fi_slow(a, b); }
#endif
static inline u128 spu_mpyu(u128 a, u128 b) { u128 r; for(int i=0;i<4;i++) r._u32[i]=(uint32_t)a._u16[i*2]*(uint32_t)b._u16[i*2]; return r; }
static inline u128 spu_mpyi(u128 a, int32_t imm) { u128 r; for(int i=0;i<4;i++) r._s32[i]=(int32_t)a._s16[i*2]*(int16_t)imm; return r; }

/* ---- bitwise logic (whole 128 bits) ---- */
static inline u128 spu_and(u128 a, u128 b) { u128 r; r._u64[0]=a._u64[0]&b._u64[0]; r._u64[1]=a._u64[1]&b._u64[1]; return r; }
static inline u128 spu_or(u128 a, u128 b)  { u128 r; r._u64[0]=a._u64[0]|b._u64[0]; r._u64[1]=a._u64[1]|b._u64[1]; return r; }
static inline u128 spu_xor(u128 a, u128 b) { u128 r; r._u64[0]=a._u64[0]^b._u64[0]; r._u64[1]=a._u64[1]^b._u64[1]; return r; }
static inline u128 spu_nand(u128 a, u128 b){ u128 r; r._u64[0]=~(a._u64[0]&b._u64[0]); r._u64[1]=~(a._u64[1]&b._u64[1]); return r; }
static inline u128 spu_nor(u128 a, u128 b) { u128 r; r._u64[0]=~(a._u64[0]|b._u64[0]); r._u64[1]=~(a._u64[1]|b._u64[1]); return r; }
static inline u128 spu_andc(u128 a, u128 b){ u128 r; r._u64[0]=a._u64[0]&~b._u64[0]; r._u64[1]=a._u64[1]&~b._u64[1]; return r; }
static inline u128 spu_orc(u128 a, u128 b) { u128 r; r._u64[0]=a._u64[0]|~b._u64[0]; r._u64[1]=a._u64[1]|~b._u64[1]; return r; }
static inline u128 spu_andi(u128 a, int32_t imm){ u128 r; for(int i=0;i<4;i++) r._u32[i]=a._u32[i]&(uint32_t)imm; return r; }
static inline u128 spu_ori(u128 a, int32_t imm) { u128 r; for(int i=0;i<4;i++) r._u32[i]=a._u32[i]|(uint32_t)imm; return r; }
static inline u128 spu_xori(u128 a, int32_t imm){ u128 r; for(int i=0;i<4;i++) r._u32[i]=a._u32[i]^(uint32_t)imm; return r; }

/* ---- count leading zeros / population count per byte ---- */
static inline u128 spu_clz(u128 a)  { u128 r; for(int i=0;i<4;i++) r._u32[i]=(uint32_t)spu_clz32(a._u32[i]); return r; }
static inline u128 spu_cntb_ref(u128 a) { u128 r; for(int i=0;i<16;i++){ uint8_t v=a._u8[i],c=0; while(v){c+=v&1;v>>=1;} r._u8[i]=c; } return r; }
#if SPU__SIMD
static inline u128 spu_cntb(u128 a){ __m128i A; memcpy(&A,&a,16);
    const __m128i lut=_mm_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4), m=_mm_set1_epi8(0x0F);
    __m128i r=_mm_add_epi8(_mm_shuffle_epi8(lut,_mm_and_si128(A,m)), _mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(A,4),m)));
    u128 o; memcpy(&o,&r,16); return o; }
#else
#define spu_cntb spu_cntb_ref
#endif

/* ---- compares: all-ones / all-zeros per lane ---- */
static inline u128 spu_ceq(u128 a, u128 b)  { u128 r; for(int i=0;i<4;i++) r._u32[i]=(a._u32[i]==b._u32[i])?0xFFFFFFFFu:0; return r; }
static inline u128 spu_ceqh(u128 a, u128 b) { u128 r; for(int i=0;i<8;i++) r._u16[i]=(a._u16[i]==b._u16[i])?0xFFFFu:0; return r; }
static inline u128 spu_ceqb(u128 a, u128 b) { u128 r; for(int i=0;i<16;i++) r._u8[i]=(a._u8[i]==b._u8[i])?0xFFu:0; return r; }
static inline u128 spu_cgt(u128 a, u128 b)  { u128 r; for(int i=0;i<4;i++) r._u32[i]=(a._s32[i]>b._s32[i])?0xFFFFFFFFu:0; return r; }
static inline u128 spu_cgth(u128 a, u128 b) { u128 r; for(int i=0;i<8;i++) r._u16[i]=(a._s16[i]>b._s16[i])?0xFFFFu:0; return r; }
static inline u128 spu_cgtb(u128 a, u128 b) { u128 r; for(int i=0;i<16;i++) r._u8[i]=(a._s8[i]>b._s8[i])?0xFFu:0; return r; }
static inline u128 spu_clgt(u128 a, u128 b) { u128 r; for(int i=0;i<4;i++) r._u32[i]=(a._u32[i]>b._u32[i])?0xFFFFFFFFu:0; return r; }
static inline u128 spu_clgth(u128 a, u128 b){ u128 r; for(int i=0;i<8;i++) r._u16[i]=(a._u16[i]>b._u16[i])?0xFFFFu:0; return r; }
static inline u128 spu_clgtb(u128 a, u128 b){ u128 r; for(int i=0;i<16;i++) r._u8[i]=(a._u8[i]>b._u8[i])?0xFFu:0; return r; }
static inline u128 spu_ceqi(u128 a, int32_t imm) { u128 r; for(int i=0;i<4;i++) r._u32[i]=(a._s32[i]==imm)?0xFFFFFFFFu:0; return r; }
static inline u128 spu_cgti(u128 a, int32_t imm) { u128 r; for(int i=0;i<4;i++) r._u32[i]=(a._s32[i]>imm)?0xFFFFFFFFu:0; return r; }
static inline u128 spu_clgti(u128 a, int32_t imm){ u128 r; for(int i=0;i<4;i++) r._u32[i]=(a._u32[i]>(uint32_t)imm)?0xFFFFFFFFu:0; return r; }

/* ---- select / shuffle ---- */
static inline u128 spu_selb(u128 a, u128 b, u128 c) {
    u128 r;
    r._u64[0]=(a._u64[0]&~c._u64[0])|(b._u64[0]&c._u64[0]);
    r._u64[1]=(a._u64[1]&~c._u64[1])|(b._u64[1]&c._u64[1]);
    return r;
}
/* shufb special selectors per Cell BE ISA:
 *   sel & 0xE0 == 0xE0 -> 0x80
 *   sel & 0xC0 == 0xC0 -> 0xFF
 *   sel & 0xC0 == 0x80 -> 0x00
 *   otherwise           -> concat{a,b}[sel & 0x1F] */
#if defined(__SSE4_1__)
#include <smmintrin.h>
/* pshufb version: one of the hottest SPU ops (~11% of GH3's FMOD mixer task
 * as the byte loop). SPU byte P lives at host byte SPU_W(P), i.e. the bytes
 * of each 32-bit word reversed, so swap into SPU order, select, swap back. */
static inline u128 spu_shufb(u128 a, u128 b, u128 c) {
    /* v131: no byte swaps. Swapping a, b, c into SPU order and the result back cancels for the control
     * (SPU_W is an involution: result host byte h is decided by control host byte h), and SPU byte j of
     * a register is host byte j^3, so the selector's low nibble indexes a/b directly after ^3. Bit 4
     * (b vs a) is moved to bit 7 for blendv by a 16-bit shift left by 3 (only each byte's bit 7 is
     * read). pshufb zeroes the 10x/110/111 bytes (bit 7 set, kept by the ^3), and a 16-entry LUT on the
     * high nibble ORs in 0x00 / 0xFF / 0x80. Everything that depends only on c hoists out of loops when
     * the control is loop-invariant (the usual case). tests/test_spu_shufb.c + v131 harness: identical
     * to spu_shufb_ref for every control byte value. */
    __m128i A, B, C; memcpy(&A, &a, 16); memcpy(&B, &b, 16); memcpy(&C, &c, 16);
    const __m128i i = _mm_xor_si128(C, _mm_set1_epi8(0x03));
    __m128i r = _mm_blendv_epi8(_mm_shuffle_epi8(A, i), _mm_shuffle_epi8(B, i), _mm_slli_epi16(C, 3));
    const __m128i lut = _mm_setr_epi8(0,0,0,0, 0,0,0,0, 0,0,0,0, (char)0xFF,(char)0xFF,(char)0x80,(char)0x80);
    r = _mm_or_si128(r, _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(C, 4), _mm_set1_epi8(0x0F))));
    u128 out; memcpy(&out, &r, 16); return out;
}
/* The byte loop stays as the reference (tests/test_spu_shufb.c checks both). */
static inline u128 spu_shufb_ref(u128 a, u128 b, u128 c) {
    /* shufb is defined on SPU byte positions. Our u128 is host-native LE, so SPU
     * byte P lives at _u8[SPU_W(P)]; map every access (the concat source, the
     * control, and the result) through SPU_W so a control supplied as an immediate
     * (ila/il) or an LS-loaded constant -- already in true SPU byte order -- is
     * interpreted correctly. The cbd/chd/cwd/cdd generators below produce true
     * SPU-byte-order selectors to match. */
    uint8_t cat[32];
    for (int j=0;j<16;j++) cat[j]    = a._u8[SPU_W(j)];   /* concat SPU byte j  = a SPU byte j */
    for (int j=0;j<16;j++) cat[16+j] = b._u8[SPU_W(j)];   /* concat SPU byte 16+j = b SPU byte j */
    u128 r;
    for (int t=0;t<16;t++) {                              /* result SPU byte t */
        uint8_t s = c._u8[SPU_W(t)];                      /* control SPU byte t */
        uint8_t v;
        if      ((s & 0xE0)==0xE0) v=0x80;
        else if ((s & 0xC0)==0xC0) v=0xFF;
        else if ((s & 0xC0)==0x80) v=0x00;
        else                       v=cat[s & 0x1F];       /* concat SPU byte (s & 0x1F) */
        r._u8[SPU_W(t)] = v;
    }
    return r;
}
#else
static inline u128 spu_shufb(u128 a, u128 b, u128 c) {
    /* shufb is defined on SPU byte positions. Our u128 is host-native LE, so SPU
     * byte P lives at _u8[SPU_W(P)]; map every access (the concat source, the
     * control, and the result) through SPU_W so a control supplied as an immediate
     * (ila/il) or an LS-loaded constant -- already in true SPU byte order -- is
     * interpreted correctly. The cbd/chd/cwd/cdd generators below produce true
     * SPU-byte-order selectors to match. */
    uint8_t cat[32];
    for (int j=0;j<16;j++) cat[j]    = a._u8[SPU_W(j)];   /* concat SPU byte j  = a SPU byte j */
    for (int j=0;j<16;j++) cat[16+j] = b._u8[SPU_W(j)];   /* concat SPU byte 16+j = b SPU byte j */
    u128 r;
    for (int t=0;t<16;t++) {                              /* result SPU byte t */
        uint8_t s = c._u8[SPU_W(t)];                      /* control SPU byte t */
        uint8_t v;
        if      ((s & 0xE0)==0xE0) v=0x80;
        else if ((s & 0xC0)==0xC0) v=0xFF;
        else if ((s & 0xC0)==0x80) v=0x00;
        else                       v=cat[s & 0x1F];       /* concat SPU byte (s & 0x1F) */
        r._u8[SPU_W(t)] = v;
    }
    return r;
}
#define spu_shufb_ref spu_shufb
#endif

/* ---- shift / rotate immediate (word lanes) ---- */
static inline u128 spu_shli(u128 a, int sh)  { u128 r; sh&=0x3F; for(int i=0;i<4;i++) r._u32[i]=(sh>31)?0:(a._u32[i]<<sh); return r; }
static inline u128 spu_shlhi(u128 a, int sh) { u128 r; sh&=0x1F; for(int i=0;i<8;i++) r._u16[i]=(sh>15)?0:(uint16_t)(a._u16[i]<<sh); return r; }
static inline u128 spu_roti(u128 a, int sh)  { u128 r; sh&=31; for(int i=0;i<4;i++) r._u32[i]= sh ? ((a._u32[i]<<sh)|(a._u32[i]>>(32-sh))) : a._u32[i]; return r; }
static inline u128 spu_rothi(u128 a, int sh) { u128 r; sh&=15; for(int i=0;i<8;i++) r._u16[i]=(uint16_t)((a._u16[i]<<sh)|(a._u16[i]>>(16-sh))); return r; }
static inline u128 spu_rotmi(u128 a, int i7)  { u128 r; int sh=(0-i7)&0x3F; for(int i=0;i<4;i++) r._u32[i]=(sh>31)?0:(a._u32[i]>>sh); return r; }
static inline u128 spu_rotmai(u128 a, int i7) { u128 r; int sh=(0-i7)&0x3F; for(int i=0;i<4;i++) r._s32[i]=(sh>31)?(a._s32[i]>>31):(a._s32[i]>>sh); return r; }
static inline u128 spu_rotmhi(u128 a, int i7) { u128 r; int sh=(0-i7)&0x1F; for(int i=0;i<8;i++) r._u16[i]=(sh>15)?0:(uint16_t)(a._u16[i]>>sh); return r; }
static inline u128 spu_shlqbyi_ref(u128 a, int sh) { u128 r=spu_zero(); sh&=0x1F; for(int i=0;i<16;i++){ int s=i+sh; if(s<16) r._u8[SPU_W(i)]=a._u8[SPU_W(s)]; } return r; }
#if SPU__SIMD
static inline u128 spu_shlqbyi(u128 a, int sh){ sh&=0x1F; return sh>=16 ? spu_zero() : spu__pshufb(a, spu__t_shl[sh]); }
#else
#define spu_shlqbyi spu_shlqbyi_ref
#endif
static inline u128 spu_rotqbyi_ref(u128 a, int sh) { u128 r; sh&=0x0F; for(int i=0;i<16;i++) r._u8[SPU_W(i)]=a._u8[SPU_W((i+sh)&0x0F)]; return r; }
#if SPU__SIMD
static inline u128 spu_rotqbyi(u128 a, int sh){ return spu__pshufb(a, spu__t_rot[sh&0x0F]); }
#else
#define spu_rotqbyi spu_rotqbyi_ref
#endif
/* Bit-level quadword shifts/rotates operate on the WHOLE 128-bit big-endian
 * value. Our _u64[0]/_u64[1] are word-SCRAMBLED (host-LE: _u64[0] = word0 |
 * word1<<32, but SPU's MS half is word0<<32 | word1), so native shifts on them
 * corrupt any bits crossing a 32-bit boundary. Assemble the logical hi/lo halves
 * with _u32[0] as most-significant, shift, then disassemble. */
static inline u128 spu_shlqbii_ref(u128 a, int sh) { sh&=7; if(!sh) return a;
    uint64_t hi=((uint64_t)a._u32[0]<<32)|a._u32[1], lo=((uint64_t)a._u32[2]<<32)|a._u32[3];
    uint64_t nhi=(hi<<sh)|(lo>>(64-sh)), nlo=(lo<<sh);
    u128 r; r._u32[0]=(uint32_t)(nhi>>32); r._u32[1]=(uint32_t)nhi; r._u32[2]=(uint32_t)(nlo>>32); r._u32[3]=(uint32_t)nlo; return r; }
static inline u128 spu_rotqbii_ref(u128 a, int sh) { sh&=7; if(!sh) return a;
    uint64_t hi=((uint64_t)a._u32[0]<<32)|a._u32[1], lo=((uint64_t)a._u32[2]<<32)|a._u32[3];
    uint64_t nhi=(hi<<sh)|(lo>>(64-sh)), nlo=(lo<<sh)|(hi>>(64-sh));
    u128 r; r._u32[0]=(uint32_t)(nhi>>32); r._u32[1]=(uint32_t)nhi; r._u32[2]=(uint32_t)(nlo>>32); r._u32[3]=(uint32_t)nlo; return r; }

/* ---- single-precision float (4 lanes) ----
 * v52 SPU_XFLOAT (approximate extended range, RPCS3's default "approximate xfloat"):
 * SPU single precision has NO Inf/NaN -- exponent 255 is just a bigger normal number
 * (up to ~6.8e38), and frest(0) / overflow produce such values on purpose. On x86 they
 * are Inf/NaN, and 0 * "max" = NaN poisons whole results (inFamous: NaN in SPU-skinned
 * character vertices, missing limb segments). Operands with exponent 255 are clamped to
 * +-FLT_MAX before every float op, so 0 * max = 0 as on the SPU. g_spu_xfloat=0 (env
 * SPU_XFLOAT=0) restores raw x86 semantics. */
extern int g_spu_xfloat;
static inline float spu_xf(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) { u = (u & 0x80000000u) | 0x7F7FFFFFu; memcpy(&x, &u, 4); }
    return x;
}
static inline u128 spu_xfv(u128 a) { if (g_spu_xfloat) for (int i = 0; i < 4; i++) a._f32[i] = spu_xf(a._f32[i]); return a; }
/* v63 (FPS): the single-precision ops as ONE packed SSE op each instead of a 4-lane scalar
 * loop (+ a branchy per-lane xfloat clamp). Bit-identical: same IEEE ops, same MXCSR, the
 * xfloat clamp is the same mask/blend; fma/fms/fnms use vfmadd/vfmsub/vfnmadd (single
 * rounding, exactly fmaf) when the TU has FMA (lifted SPU sources: -mfma), else fmaf. */
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#if defined(__FMA__)
#include <immintrin.h>
#endif
#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif
#define SPU__FSIMD 1
static inline __m128 spu__fld(u128 a) { __m128 v; memcpy(&v, &a, 16); return v; }
static inline u128 spu__fst(__m128 v) { u128 r; memcpy(&r, &v, 16); return r; }
#if defined(__SSE4_1__)
/* v131 (FPS): the same clamp as two unsigned/signed integer mins against per-run limits instead of a
 * mask/blend behind an `if (g_spu_xfloat)` branch. Every float op in a lifted kernel used to carry that
 * branch (clang does not unswitch the big SPU loops), which also cost spills around each op.
 * Positive bits 0x7F800000..0x7FFFFFFF -> 0x7F7FFFFF (signed min), negative 0xFF800000..0xFFFFFFFF ->
 * 0xFF7FFFFF (unsigned min), everything else unchanged == the blend version bit for bit (exhaustively
 * tested over all 2^32 patterns). g_spu_xflim = {0xFF7FFFFF x4, 0x7F7FFFFF x4}, or all-pass limits
 * {0xFFFFFFFF x4, 0x7FFFFFFF x4} when SPU_XFLOAT=0 (spu_interp.c keeps it in sync with g_spu_xfloat). */
extern uint32_t g_spu_xflim[8];
static inline __m128 spu__xfq(__m128 v)
{
    __m128i x = _mm_castps_si128(v);
#if defined(SPU_RC_XFLOAT_FIXED)
    /* Experimental RC-only specialization. The registry serves the original
     * lifted function when SPU_XFLOAT=0, so the limits are fixed for this TU. */
    x = _mm_min_epu32(x, _mm_set1_epi32((int)0xFF7FFFFFu));
    x = _mm_min_epi32(x, _mm_set1_epi32(0x7F7FFFFF));
#else
    x = _mm_min_epu32(x, _mm_loadu_si128((const __m128i*)&g_spu_xflim[0]));
    x = _mm_min_epi32(x, _mm_loadu_si128((const __m128i*)&g_spu_xflim[4]));
#endif
    return _mm_castsi128_ps(x);
}
#else
static inline __m128 spu__xfq(__m128 v)
{
    if (!g_spu_xfloat) return v;
    const __m128i e = _mm_set1_epi32(0x7F800000);
    __m128i i = _mm_castps_si128(v);
    __m128i m = _mm_cmpeq_epi32(_mm_and_si128(i, e), e);
    __m128i cl = _mm_or_si128(_mm_and_si128(i, _mm_set1_epi32((int)0x80000000u)), _mm_set1_epi32(0x7F7FFFFF));
    return _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(m, cl), _mm_andnot_si128(m, i)));
}
#endif
static inline u128 spu_fa(u128 a, u128 b) { return spu__fst(_mm_add_ps(spu__xfq(spu__fld(a)), spu__xfq(spu__fld(b)))); }
static inline u128 spu_fs(u128 a, u128 b) { return spu__fst(_mm_sub_ps(spu__xfq(spu__fld(a)), spu__xfq(spu__fld(b)))); }
static inline u128 spu_fm(u128 a, u128 b) { return spu__fst(_mm_mul_ps(spu__xfq(spu__fld(a)), spu__xfq(spu__fld(b)))); }
#if defined(__FMA__)
static inline u128 spu_fma(u128 a, u128 b, u128 c)  { return spu__fst(_mm_fmadd_ps (spu__xfq(spu__fld(a)), spu__xfq(spu__fld(b)), spu__xfq(spu__fld(c)))); }
static inline u128 spu_fms(u128 a, u128 b, u128 c)  { return spu__fst(_mm_fmsub_ps (spu__xfq(spu__fld(a)), spu__xfq(spu__fld(b)), spu__xfq(spu__fld(c)))); }
static inline u128 spu_fnms(u128 a, u128 b, u128 c) { return spu__fst(_mm_fnmadd_ps(spu__xfq(spu__fld(a)), spu__xfq(spu__fld(b)), spu__xfq(spu__fld(c)))); }
#else
static inline u128 spu_fma(u128 a, u128 b, u128 c)  { u128 r; a=spu__fst(spu__xfq(spu__fld(a))); b=spu__fst(spu__xfq(spu__fld(b))); c=spu__fst(spu__xfq(spu__fld(c))); for(int i=0;i<4;i++) r._f32[i]=fmaf(a._f32[i], b._f32[i], c._f32[i]); return r; }
static inline u128 spu_fms(u128 a, u128 b, u128 c)  { u128 r; a=spu__fst(spu__xfq(spu__fld(a))); b=spu__fst(spu__xfq(spu__fld(b))); c=spu__fst(spu__xfq(spu__fld(c))); for(int i=0;i<4;i++) r._f32[i]=fmaf(a._f32[i], b._f32[i], -c._f32[i]); return r; }
static inline u128 spu_fnms(u128 a, u128 b, u128 c) { u128 r; a=spu__fst(spu__xfq(spu__fld(a))); b=spu__fst(spu__xfq(spu__fld(b))); c=spu__fst(spu__xfq(spu__fld(c))); for(int i=0;i<4;i++) r._f32[i]=fmaf(a._f32[i], -b._f32[i], c._f32[i]); return r; }
#endif
#else
#define SPU__FSIMD 0
#endif
#if !SPU__FSIMD
static inline u128 spu_fa(u128 a, u128 b) { u128 r; a=spu_xfv(a); b=spu_xfv(b); for(int i=0;i<4;i++) r._f32[i]=a._f32[i]+b._f32[i]; return r; }
#endif
#if !SPU__FSIMD
static inline u128 spu_fs(u128 a, u128 b) { u128 r; a=spu_xfv(a); b=spu_xfv(b); for(int i=0;i<4;i++) r._f32[i]=a._f32[i]-b._f32[i]; return r; }
#endif

/* v63: exact 2^e for the RI8 scales below (was a libm exp2() call per instruction). */
static inline double spu__p2(int e) { uint64_t u = (uint64_t)(1023 + e) << 52; double d; memcpy(&d, &u, 8); return d; }
/* RI8 scale: float-to-int uses 2^(173-i8), int-to-float 2^(i8-155).
 * The canonical immediates 173 and 155 are identity conversions. */
static inline u128 spu_cflts_ref(u128 a, int i8){ u128 r; double f=spu__p2(173-i8); a=spu_xfv(a);
    for(int i=0;i<4;i++){ double v=(double)a._f32[i]*f; if(v>2147483647.0)v=2147483647.0; if(v<-2147483648.0)v=-2147483648.0; r._s32[i]=(int32_t)v; } return r; }
static inline u128 spu_cfltu_ref(u128 a, int i8){ u128 r; double f=spu__p2(173-i8); a=spu_xfv(a);
    for(int i=0;i<4;i++){ double v=(double)a._f32[i]*f; if(v<0)v=0; if(v>4294967295.0)v=4294967295.0; r._u32[i]=(uint32_t)v; } return r; }
static inline u128 spu_csflt_ref(u128 a, int i8){ u128 r; double f=spu__p2(i8-155);
    for(int i=0;i<4;i++) r._f32[i]=(float)((double)a._s32[i]*f); return r; }
static inline u128 spu_cuflt_ref(u128 a, int i8){ u128 r; double f=spu__p2(i8-155);
    for(int i=0;i<4;i++) r._f32[i]=(float)((double)a._u32[i]*f); return r; }
#if !SPU__FSIMD
static inline u128 spu_fm(u128 a, u128 b) { u128 r; a=spu_xfv(a); b=spu_xfv(b); for(int i=0;i<4;i++) r._f32[i]=a._f32[i]*b._f32[i]; return r; }
#endif
#if !SPU__FSIMD
static inline u128 spu_fma(u128 a, u128 b, u128 c)  { u128 r; a=spu_xfv(a); b=spu_xfv(b); c=spu_xfv(c); for(int i=0;i<4;i++) r._f32[i]=fmaf(a._f32[i], b._f32[i], c._f32[i]); return r; }
#endif
#if !SPU__FSIMD
static inline u128 spu_fms(u128 a, u128 b, u128 c)  { u128 r; a=spu_xfv(a); b=spu_xfv(b); c=spu_xfv(c); for(int i=0;i<4;i++) r._f32[i]=fmaf(a._f32[i], b._f32[i], -c._f32[i]); return r; }
#endif
#if !SPU__FSIMD
static inline u128 spu_fnms(u128 a, u128 b, u128 c) { u128 r; a=spu_xfv(a); b=spu_xfv(b); c=spu_xfv(c); for(int i=0;i<4;i++) r._f32[i]=fmaf(a._f32[i], -b._f32[i], c._f32[i]); return r; }
#endif
static inline u128 spu_fceq_ref(u128 a, u128 b) { u128 r; a=spu_xfv(a); b=spu_xfv(b); for(int i=0;i<4;i++) r._u32[i]=(a._f32[i]==b._f32[i])?0xFFFFFFFFu:0; return r; }
static inline u128 spu_fcgt_ref(u128 a, u128 b) { u128 r; a=spu_xfv(a); b=spu_xfv(b); for(int i=0;i<4;i++) r._u32[i]=(a._f32[i]>b._f32[i])?0xFFFFFFFFu:0; return r; }

/* ---- SPU double-precision (2 doubles/reg; dword i = words 2i(high),2i+1(low)).
 * Reassemble via _u32 -- our u128 is host-LE, so a naive _f64[i] would SWAP the
 * two words. Semantics from RPCS3 SPUInterpreter (DFASM / DFMA / FESD / FRDS). ---- */
/* SPU DP is IEEE round-to-nearest with denormals; SPU SP runs with MXCSR =
 * truncate + FTZ/DAZ (SPU_FLOAT_MODE). Switch per DP op HERE so lifted code and
 * the interpreter agree (2026-10-04: lifted WWS jobs call these directly). */
#include <xmmintrin.h>
#define SPU__DP_BEGIN { unsigned _spu_dpc = _mm_getcsr(); _mm_setcsr(_spu_dpc & ~0xE040u);
#define SPU__DP_END   _mm_setcsr(_spu_dpc); }
static inline double spu__dget(u128 r, int i) {
    uint64_t u = ((uint64_t)r._u32[i*2] << 32) | r._u32[i*2+1];
    double d; memcpy(&d, &u, sizeof d); return d;
}
static inline void spu__dset(u128* r, int i, double d) {
    uint64_t u; memcpy(&u, &d, sizeof u);
    r->_u32[i*2] = (uint32_t)(u >> 32); r->_u32[i*2+1] = (uint32_t)u;
}
static inline u128 spu_dfa(u128 a, u128 b) { u128 r; SPU__DP_BEGIN for(int i=0;i<2;i++) spu__dset(&r,i, spu__dget(a,i)+spu__dget(b,i)); SPU__DP_END return r; }
static inline u128 spu_dfs(u128 a, u128 b) { u128 r; SPU__DP_BEGIN for(int i=0;i<2;i++) spu__dset(&r,i, spu__dget(a,i)-spu__dget(b,i)); SPU__DP_END return r; }
static inline u128 spu_dfm(u128 a, u128 b) { u128 r; SPU__DP_BEGIN for(int i=0;i<2;i++) spu__dset(&r,i, spu__dget(a,i)*spu__dget(b,i)); SPU__DP_END return r; }
/* double FMA family: c = rt (accumulator, 3-register). */
static inline u128 spu_dfma(u128 a, u128 b, u128 t) { u128 r; SPU__DP_BEGIN for(int i=0;i<2;i++) spu__dset(&r,i, spu__dget(a,i)*spu__dget(b,i)+spu__dget(t,i)); SPU__DP_END return r; }
static inline u128 spu_dfms(u128 a, u128 b, u128 t) { u128 r; SPU__DP_BEGIN for(int i=0;i<2;i++) spu__dset(&r,i, spu__dget(a,i)*spu__dget(b,i)-spu__dget(t,i)); SPU__DP_END return r; }
static inline u128 spu_dfnms(u128 a, u128 b, u128 t) { u128 r; SPU__DP_BEGIN for(int i=0;i<2;i++) spu__dset(&r,i, spu__dget(t,i)-spu__dget(a,i)*spu__dget(b,i)); SPU__DP_END return r; }
static inline u128 spu_dfnma(u128 a, u128 b, u128 t) { u128 r; SPU__DP_BEGIN for(int i=0;i<2;i++) spu__dset(&r,i, -(spu__dget(a,i)*spu__dget(b,i)+spu__dget(t,i))); SPU__DP_END return r; }
/* double compares -> per-lane 64-bit mask (RPCS3 stubs these; sane impl here). */
static inline u128 spu_dfceq(u128 a, u128 b)  { u128 r; for(int i=0;i<2;i++){ uint64_t m=(spu__dget(a,i)==spu__dget(b,i))?~0ull:0ull; r._u32[i*2]=(uint32_t)(m>>32); r._u32[i*2+1]=(uint32_t)m; } return r; }
static inline u128 spu_dfcmeq(u128 a, u128 b) { u128 r; for(int i=0;i<2;i++){ double x=spu__dget(a,i),y=spu__dget(b,i); if(x<0)x=-x; if(y<0)y=-y; uint64_t m=(x==y)?~0ull:0ull; r._u32[i*2]=(uint32_t)(m>>32); r._u32[i*2+1]=(uint32_t)m; } return r; }
/* Singles occupy preferred words 0 and 2 when converting doubleword lanes. */
static inline u128 spu_fesd(u128 a) { u128 r; memset(&r,0,sizeof r); for(int i=0;i<2;i++) spu__dset(&r,i,(double)a._f32[i*2]); return r; }
static inline u128 spu_frds(u128 a) { u128 r; memset(&r,0,sizeof r); for(int i=0;i<2;i++) r._f32[i*2]=(float)spu__dget(a,i); return r; }
/* mpyhhu: high-16 x high-16 of each word, unsigned, full 32-bit product. */
static inline u128 spu_mpyhhu(u128 a, u128 b){ u128 r; for(int i=0;i<4;i++) r._u32[i]=(uint32_t)a._u16[2*i+1]*(uint32_t)b._u16[2*i+1]; return r; }
/* cgx: extended carry-generate, carry-in = low bit of old rt (RPCS3 CGX). */
static inline u128 spu_cgx(u128 a, u128 b, u128 t){ u128 r; for(int i=0;i<4;i++) r._u32[i]=(uint32_t)(((uint64_t)(t._u32[i]&1u)+a._u32[i]+b._u32[i])>>32); return r; }

/* ---- remaining SPU ISA ops (RPCS3 SPUInterpreter: EQV/ABSDB/AVGB/MPYHHA/
 * MPYHHAU/DFCGT/DFCMGT/XORBI/DFTSV). Completes the lifter's opcode coverage. ---- */
static inline u128 spu_eqv(u128 a, u128 b)   { u128 r; for(int i=0;i<4;i++) r._u32[i]=~(a._u32[i]^b._u32[i]); return r; }
static inline u128 spu_xorbi(u128 a, int32_t imm){ u128 r; for(int i=0;i<16;i++) r._u8[i]=(uint8_t)(a._u8[i]^(uint8_t)imm); return r; }
static inline u128 spu_absdb(u128 a, u128 b)  { u128 r; for(int i=0;i<16;i++){ uint8_t x=a._u8[i],y=b._u8[i]; r._u8[i]=(uint8_t)(x>y?x-y:y-x); } return r; }
static inline u128 spu_avgb(u128 a, u128 b)   { u128 r; for(int i=0;i<16;i++) r._u8[i]=(uint8_t)(((uint32_t)a._u8[i]+(uint32_t)b._u8[i]+1u)>>1); return r; }
/* mpyhha/mpyhhau: high-16 x high-16, ACCUMULATE into rt (3-register). */
static inline u128 spu_mpyhha(u128 a, u128 b, u128 t)  { u128 r; for(int i=0;i<4;i++) r._s32[i]=t._s32[i]+(int32_t)a._s16[2*i+1]*(int32_t)b._s16[2*i+1]; return r; }
static inline u128 spu_mpyhhau(u128 a, u128 b, u128 t) { u128 r; for(int i=0;i<4;i++) r._u32[i]=t._u32[i]+(uint32_t)a._u16[2*i+1]*(uint32_t)b._u16[2*i+1]; return r; }
/* double compare greater (RPCS3 stubs these; sane impl) -> per-lane mask. */
static inline u128 spu_dfcgt(u128 a, u128 b)  { u128 r; for(int i=0;i<2;i++){ uint64_t m=(spu__dget(a,i)>spu__dget(b,i))?~0ull:0ull; r._u32[i*2]=(uint32_t)(m>>32); r._u32[i*2+1]=(uint32_t)m; } return r; }
static inline u128 spu_dfcmgt(u128 a, u128 b) { u128 r; for(int i=0;i<2;i++){ double x=spu__dget(a,i),y=spu__dget(b,i); if(x<0)x=-x; if(y<0)y=-y; uint64_t m=(x>y)?~0ull:0ull; r._u32[i*2]=(uint32_t)(m>>32); r._u32[i*2+1]=(uint32_t)m; } return r; }
/* dftsv: double test special value -- RPCS3 stubs (fatal); 0 = no special. */
static inline u128 spu_dftsv(u128 a, int32_t imm){ u128 r; for(int i=0;i<2;i++){ uint64_t v=((uint64_t)a._u32[i*2]<<32)|a._u32[i*2+1]; int neg=(int)(v>>63); uint64_t e=(v>>52)&0x7FF, f=v&0xFFFFFFFFFFFFFull; int hit=0; if(e==0x7FF&&f) hit=(imm&0x40)!=0; else if(e==0x7FF) hit=(imm&(neg?0x10:0x20))!=0; else if(e==0&&!f) hit=(imm&(neg?0x04:0x08))!=0; else if(e==0) hit=(imm&(neg?0x01:0x02))!=0; uint32_t m=hit?~0u:0u; r._u32[i*2]=m; r._u32[i*2+1]=m; } return r; }

/* ---- Phase 2: register-variable shifts/rotates ---- */
static inline u128 spu_shl(u128 a, u128 b)   { u128 r; for(int i=0;i<4;i++){ uint32_t sh=b._u32[i]&0x3F; r._u32[i]=(sh>31)?0:(a._u32[i]<<sh); } return r; }
static inline u128 spu_shlh(u128 a, u128 b)  { u128 r; for(int i=0;i<8;i++){ uint32_t sh=b._u16[i]&0x1F; r._u16[i]=(sh>15)?0:(uint16_t)(a._u16[i]<<sh); } return r; }
static inline u128 spu_rot(u128 a, u128 b)   { u128 r; for(int i=0;i<4;i++){ uint32_t sh=b._u32[i]&31; r._u32[i]= sh ? ((a._u32[i]<<sh)|(a._u32[i]>>(32-sh))) : a._u32[i]; } return r; }
static inline u128 spu_roth(u128 a, u128 b)  { u128 r; for(int i=0;i<8;i++){ uint32_t sh=b._u16[i]&15; r._u16[i]= sh ? (uint16_t)((a._u16[i]<<sh)|(a._u16[i]>>(16-sh))) : a._u16[i]; } return r; }
static inline u128 spu_shlqbi_ref(u128 a, u128 b){ int sh=b._u32[0]&7; if(!sh) return a;
    uint64_t hi=((uint64_t)a._u32[0]<<32)|a._u32[1], lo=((uint64_t)a._u32[2]<<32)|a._u32[3];
    uint64_t nhi=(hi<<sh)|(lo>>(64-sh)), nlo=(lo<<sh);
    u128 r; r._u32[0]=(uint32_t)(nhi>>32); r._u32[1]=(uint32_t)nhi; r._u32[2]=(uint32_t)(nlo>>32); r._u32[3]=(uint32_t)nlo; return r; }
static inline u128 spu_rotqbi_ref(u128 a, u128 b){ int sh=b._u32[0]&7; if(!sh) return a;
    uint64_t hi=((uint64_t)a._u32[0]<<32)|a._u32[1], lo=((uint64_t)a._u32[2]<<32)|a._u32[3];
    uint64_t nhi=(hi<<sh)|(lo>>(64-sh)), nlo=(lo<<sh)|(hi>>(64-sh));
    u128 r; r._u32[0]=(uint32_t)(nhi>>32); r._u32[1]=(uint32_t)nhi; r._u32[2]=(uint32_t)(nlo>>32); r._u32[3]=(uint32_t)nlo; return r; }
static inline u128 spu_shlqby_ref(u128 a, u128 b){ int sh=b._u32[0]&0x1F; u128 r=spu_zero(); if(sh>=16) return r; for(int i=0;i<16;i++){ int s=i+sh; if(s<16) r._u8[SPU_W(i)]=a._u8[SPU_W(s)]; } return r; }
#if SPU__SIMD
static inline u128 spu_shlqby(u128 a, u128 b){ int sh=b._u32[0]&0x1F; return sh>=16 ? spu_zero() : spu__pshufb(a, spu__t_shl[sh]); }
#else
#define spu_shlqby spu_shlqby_ref
#endif
static inline u128 spu_rotqby_ref(u128 a, u128 b){ int sh=b._u32[0]&0x0F; u128 r; for(int i=0;i<16;i++) r._u8[SPU_W(i)]=a._u8[SPU_W((i+sh)&0x0F)]; return r; }
#if SPU__SIMD
static inline u128 spu_rotqby(u128 a, u128 b){ return spu__pshufb(a, spu__t_rot[b._u32[0]&0x0F]); }
#else
#define spu_rotqby spu_rotqby_ref
#endif
static inline u128 spu_shlqbybi_ref(u128 a, u128 b){ int sh=(b._u32[0]>>3)&0x1F; u128 r=spu_zero(); if(sh>=16) return r; for(int i=0;i<16;i++){ int s=i+sh; if(s<16) r._u8[SPU_W(i)]=a._u8[SPU_W(s)]; } return r; }
#if SPU__SIMD
static inline u128 spu_shlqbybi(u128 a, u128 b){ int sh=(b._u32[0]>>3)&0x1F; return sh>=16 ? spu_zero() : spu__pshufb(a, spu__t_shl[sh]); }
#else
#define spu_shlqbybi spu_shlqbybi_ref
#endif
static inline u128 spu_rotqbybi_ref(u128 a, u128 b){ int sh=(b._u32[0]>>3)&0x0F; u128 r; for(int i=0;i<16;i++) r._u8[SPU_W(i)]=a._u8[SPU_W((i+sh)&0x0F)]; return r; }
#if SPU__SIMD
static inline u128 spu_rotqbybi(u128 a, u128 b){ return spu__pshufb(a, spu__t_rot[(b._u32[0]>>3)&0x0F]); }
#else
#define spu_rotqbybi spu_rotqbybi_ref
#endif

/* ---- Phase 2: rotmahi ---- */
static inline u128 spu_rotmahi(u128 a, int i7) { u128 r; int sh=(0-i7)&0x1F; for(int i=0;i<8;i++) r._s16[i]=(sh>15)?(a._s16[i]>>15):(a._s16[i]>>sh); return r; }

/* ---- Phase 2: byte/half immediate compares ---- */
static inline u128 spu_ceqbi(u128 a, int32_t imm)  { u128 r; uint8_t v=(uint8_t)imm; for(int i=0;i<16;i++) r._u8[i]=(a._u8[i]==v)?0xFFu:0; return r; }
static inline u128 spu_ceqhi(u128 a, int32_t imm)  { u128 r; int16_t v=(int16_t)imm; for(int i=0;i<8;i++) r._u16[i]=(a._s16[i]==v)?0xFFFFu:0; return r; }
static inline u128 spu_clgtbi(u128 a, int32_t imm) { u128 r; uint8_t v=(uint8_t)imm; for(int i=0;i<16;i++) r._u8[i]=(a._u8[i]>v)?0xFFu:0; return r; }
static inline u128 spu_clgthi(u128 a, int32_t imm) { u128 r; uint16_t v=(uint16_t)imm; for(int i=0;i<8;i++) r._u16[i]=(a._u16[i]>v)?0xFFFFu:0; return r; }
static inline u128 spu_cgthi(u128 a, int32_t imm)  { u128 r; int16_t v=(int16_t)imm; for(int i=0;i<8;i++) r._u16[i]=(a._s16[i]>v)?0xFFFFu:0; return r; }
static inline u128 spu_cgtbi(u128 a, int32_t imm)  { u128 r; int8_t v=(int8_t)imm; for(int i=0;i<16;i++) r._u8[i]=(a._s8[i]>v)?0xFFu:0; return r; }

/* ---- Phase 2: misc one-offs ---- */
static inline u128 spu_fscrrd(u128 a) { (void)a; return spu_zero(); }
static inline u128 spu_gb(u128 a) {
    uint32_t v = ((a._u32[0]&1)<<3)|((a._u32[1]&1)<<2)|((a._u32[2]&1)<<1)|(a._u32[3]&1);
    u128 r = spu_zero(); r._u32[0]=v; return r;
}
static inline u128 spu_gbh(u128 a) {
    /* gather LSB of each SPU halfword H into bit (7-H). SPU halfword H maps to
     * our _u16[H^1] (the within-word halfword swap of the value layout). */
    uint32_t v=0; for(int i=0;i<8;i++) v |= ((uint32_t)(a._u16[i^1]&1) << (7-i));
    u128 r = spu_zero(); r._u32[0]=v; return r;
}
/* gather LSB of each SPU byte i into bit (15-i); exact inverse of spu_fsmb.
 * SPU byte i lives at our _u8[SPU_W(i)] (byte-reversed within each word). */
static inline u128 spu_gbb(u128 a) {
    uint32_t v=0; for(int i=0;i<16;i++) v |= ((uint32_t)(a._u8[SPU_W(i)]&1) << (15-i));
    u128 r = spu_zero(); r._u32[0]=v; return r;
}
static inline u128 spu_cg(u128 a, u128 b)   { u128 r; for(int i=0;i<4;i++) r._u32[i]=(uint32_t)(((uint64_t)a._u32[i]+(uint64_t)b._u32[i])>>32); return r; }
/* sumb: sum the 4 bytes of each word. Per CBEA, RT halfword 2i (the HIGH half of
 * word i) = sum of RB's 4 bytes of word i; halfword 2i+1 (LOW half) = sum of RA's
 * 4 bytes of word i. Computed on word VALUES so byte order is irrelevant. */
static inline u128 spu_sumb(u128 a, u128 b) {
    u128 r;
    for(int i=0;i<4;i++) {
        uint32_t wa=a._u32[i], wb=b._u32[i];
        uint32_t sa=((wa>>24)&0xFF)+((wa>>16)&0xFF)+((wa>>8)&0xFF)+(wa&0xFF);
        uint32_t sb=((wb>>24)&0xFF)+((wb>>16)&0xFF)+((wb>>8)&0xFF)+(wb&0xFF);
        r._u32[i]=(sb<<16)|sa;
    }
    return r;
}
/* Borrow generate: carry-out of (b + ~a + 1) == (b >= a unsigned ? 1 : 0).
 * The subtract-side sibling of cg; pairs with sf/sfx for extended subtraction. */
static inline u128 spu_bg(u128 a, u128 b)   { u128 r; for(int i=0;i<4;i++) r._u32[i]=(uint32_t)(((uint64_t)b._u32[i]+(uint64_t)(~a._u32[i])+1u)>>32); return r; }
static inline u128 spu_addx(u128 a, u128 b, u128 t) { u128 r; for(int i=0;i<4;i++) r._u32[i]=a._u32[i]+b._u32[i]+(t._u32[i]&1); return r; }
/* LE host: high half of word i = _s16[2i+1], low half = _s16[2i]. */
static inline u128 spu_mpyh(u128 a, u128 b) { u128 r; for(int i=0;i<4;i++) r._s32[i]=((int32_t)a._s16[2*i+1] * (int32_t)b._s16[2*i]) << 16; return r; }
static inline u128 spu_mpyhh(u128 a, u128 b){ u128 r; for(int i=0;i<4;i++) r._s32[i]=(int32_t)a._s16[2*i+1] * (int32_t)b._s16[2*i+1]; return r; }
static inline u128 spu_mpys(u128 a, u128 b) { u128 r; for(int i=0;i<4;i++){ int32_t p=(int32_t)a._s16[2*i]*(int32_t)b._s16[2*i]; r._s32[i]=(int16_t)(p>>16); } return r; }
static inline u128 spu_mpyui(u128 a, int32_t imm) { u128 r; for(int i=0;i<4;i++) r._u32[i]=(uint32_t)a._u16[2*i]*(uint32_t)(uint16_t)imm; return r; }
static inline u128 spu_fcmeq_ref(u128 a, u128 b){ u128 r; a=spu_xfv(a); b=spu_xfv(b); for(int i=0;i<4;i++){ float fa=fabsf(a._f32[i]),fb=fabsf(b._f32[i]); r._u32[i]=(fa==fb)?0xFFFFFFFFu:0; } return r; }
static inline u128 spu_fcmgt_ref(u128 a, u128 b){ u128 r; a=spu_xfv(a); b=spu_xfv(b); for(int i=0;i<4;i++){ float fa=fabsf(a._f32[i]),fb=fabsf(b._f32[i]); r._u32[i]=(fa>fb)?0xFFFFFFFFu:0; } return r; }
static const uint32_t spu_frest_fraction_lut[32] = {
    0x7FFBE0, 0x7F87A6, 0x70EF72, 0x708B40, 0x638B12, 0x633AEA, 0x5792C4, 0x574AA0,
    0x4CCA7E, 0x4C9262, 0x430A44, 0x42D62A, 0x3A2E12, 0x39FDFA, 0x3215E4, 0x31F1D2,
    0x2AA9BE, 0x2A85AC, 0x23D59A, 0x23BD8E, 0x1D8576, 0x1D8576, 0x17AD5A, 0x17AD5A,
    0x124543, 0x124543, 0x0D392D, 0x0D392D, 0x08851A, 0x08851A, 0x041D07, 0x041D07
};

static const uint32_t spu_frsqest_fraction_lut[64] = {
    0x350160, 0x34E954, 0x2F993D, 0x2F993D, 0x2AA523, 0x2AA523, 0x26190D, 0x26190D,
    0x21E4F9, 0x21E4F9, 0x1E00E9, 0x1E00E9, 0x1A5CD9, 0x1A5CD9, 0x16F8CB, 0x16F8CB,
    0x13CCC0, 0x13CCC0, 0x10CCB3, 0x10CCB3, 0x0E00AA, 0x0E00AA, 0x0B58A1, 0x0B58A1,
    0x08D498, 0x08D498, 0x067491, 0x067491, 0x043089, 0x043089, 0x020C83, 0x020C83,
    0x7FFDF4, 0x7FD1DE, 0x7859C8, 0x783DBA, 0x71559C, 0x71559C, 0x6AE57C, 0x6AE57C,
    0x64F561, 0x64F561, 0x5F7149, 0x5F7149, 0x5A4D33, 0x5A4D33, 0x55811F, 0x55811F,
    0x51050F, 0x51050F, 0x4CC8FE, 0x4CC8FE, 0x48D0F0, 0x48D0F0, 0x4510E4, 0x4510E4,
    0x4180D7, 0x4180D7, 0x3E24CC, 0x3E24CC, 0x3AF4C3, 0x3AF4C3, 0x37E8BA, 0x37E8BA
};

static inline u128 spu_frsqest(u128 a) {
    u128 r;
    for (int i = 0; i < 4; i++) {
        uint32_t bits = a._u32[i];
        uint32_t exp  = (bits >> 23) & 0xFFu;
        uint32_t frac_idx = (bits >> 18) & 0x3Fu;
        uint32_t rexp = (exp == 0u) ? 0xFFu : (190u - (exp + 1u) / 2u);
        uint32_t fraction = spu_frsqest_fraction_lut[frac_idx];
        uint32_t out = (rexp << 23) | (fraction & 0x7FFFFFu);
        memcpy(&r._f32[i], &out, sizeof out);
    }
    return r;
}
/* frest: reciprocal estimate (refined by the following spu_fi Newton step).
 * Full-precision 1/x is exact after fi and >= HW-estimate accuracy. */
static inline u128 spu_frest(u128 a) {
    u128 r;
    for (int i = 0; i < 4; i++) {
        uint32_t bits = a._u32[i];
        uint32_t sign = bits & 0x80000000u;
        uint32_t exp  = (bits >> 23) & 0xFFu;
        uint32_t frac_idx = (bits >> 18) & 0x1Fu;
        uint32_t out;
        if (exp == 0u) {
            /* zero or denormal operand: ISA p.215 "1/0 is defined to give the
             * maximum SPU single-precision extended-range fp number", sign of
             * the (zero) input preserved. Same bucket for true denormals
             * (their biased exponent field is also 0). */
            out = sign | 0x7FFFFFFFu;
        } else {
            /* result biased exponent = 253 - input biased exponent (253 =
             * 2*127-1, i.e. reciprocal negates the unbiased exponent),
             * clamped to 0 on underflow (ISA p.215 "Big: if |x|>=2^126,
             * 1/x underflows to zero"). Verified bit-exact against RPCS3's
             * raw 256-entry spu_frest_exponent_lut for all 256 inputs. */
            uint32_t rexp = (exp >= 253u) ? 0u : (253u - exp);
            uint32_t fraction = spu_frest_fraction_lut[frac_idx];
            out = sign | (rexp << 23) | (fraction & 0x7FFFFFu);
        }
        memcpy(&r._f32[i], &out, sizeof out);
    }
    return r;
}

/* ---- Phase 3: sign extension ----
 * LE host: low sub-lane = _u8[2i] / _s16[2i] / _s32[2i] (the byte/half/word
 * at the *lower* storage address). Same caveat as the mpy family. */
static inline u128 spu_xsbh(u128 a) { u128 r; for(int i=0;i<8;i++) r._s16[i] = (int8_t)a._u8[2*i]; return r; }
static inline u128 spu_xshw(u128 a) { u128 r; for(int i=0;i<4;i++) r._s32[i] = (int16_t)a._s16[2*i]; return r; }
/* xswd: rt.doubleword[d] = sign_extend_64(ra.word[2d+1]) -- the ODD word of
 * each doubleword, because that is the low half in big-endian order.
 *
 * This read the EVEN words (2d) and wrote the result through _s64, which is a
 * host little-endian lane: writing it puts the value in _u32[2d] (SPU word 2d,
 * the HIGH half) and the sign fill in _u32[2d+1]. Both halves were therefore
 * wrong, and the common case -- a small positive value -- came out as ZERO.
 *
 * You Don't Know Jack's FMOD mixer hit exactly that: a compiler-generated
 * 64-bit multiply helper received `1 * 0` instead of `1 * 16`, so the mixer's
 * block count was zero and it stopped on its own assert. Unlike its siblings
 * xsbh/xshw, whose source and destination swaps cancel, nothing cancels here.
 * Write the two words explicitly. */
static inline u128 spu_xswd(u128 a) {
    u128 r;
    for (int d = 0; d < 2; d++) {
        int32_t v = (int32_t)a._u32[2*d + 1];
        r._u32[2*d]     = (uint32_t)(v >> 31);   /* sign fill (high word) */
        r._u32[2*d + 1] = (uint32_t)v;           /* value      (low word) */
    }
    return r;
}

/* ---- Phase 3: OR across ---- */
static inline u128 spu_orx(u128 a) {
    u128 r = spu_zero();
    r._u32[0] = a._u32[0] | a._u32[1] | a._u32[2] | a._u32[3];
    return r;
}

/* ---- Phase 3: form-select mask from bits ---- */
static inline u128 spu_fsm(u128 a) {
    u128 r; uint32_t v = a._u32[0] & 0xF;
    for(int i=0;i<4;i++) r._u32[i] = ((v>>(3-i))&1) ? 0xFFFFFFFFu : 0;
    return r;
}
/* fsmh: per-halfword mask from 8 bits. SPU halfword H <- bit (7-H); store at
 * our _u16[H^1] (within-word halfword swap of the value layout). Per-halfword
 * both bytes are identical so byte order within a halfword is irrelevant, but
 * the halfword ORDER within each word must be swapped. */
static inline u128 spu_fsmh(u128 a) {
    u128 r; uint32_t v = a._u32[0] & 0xFF;
    for(int i=0;i<8;i++) r._u16[i^1] = ((v>>(7-i))&1) ? 0xFFFFu : 0;
    return r;
}
/* fsmb/fsmbi: per-byte mask from 16 bits. SPU byte position P <- bit (15-P);
 * store at our _u8[SPU_W(P)] (byte-reversed within each word). A bit set at SPU
 * byte P must land at our _u8[SPU_W(P)] so that downstream word arithmetic (e.g.
 * andbi(fsmbi(0x101),-128) building a +0x80 EA offset in the low byte of words
 * 1,3) places the value in the correct byte lane. Raw _u8[P] would put 0x80 in
 * the high byte (bit 31) instead -> wrong DMA EA. */
static inline u128 spu_fsmb(u128 a) {
    u128 r; uint32_t v = a._u32[0] & 0xFFFF;
    for(int i=0;i<16;i++) r._u8[SPU_W(i)] = ((v>>(15-i))&1) ? 0xFFu : 0;
    return r;
}
static inline u128 spu_fsmbi(int32_t imm) {
    u128 r; uint32_t v = imm & 0xFFFF;
    for(int i=0;i<16;i++) r._u8[SPU_W(i)] = ((v>>(15-i))&1) ? 0xFFu : 0;
    return r;
}

/* ---- Phase 3: constant generators (insertion shuffle patterns) ---- */
/* Generate-controls (cbd/chd/cwd/cdd) — insertion selectors for the (now
 * SPU-byte-correct) shufb. Each builds a control in TRUE SPU byte order: the
 * base selects b's SPU byte t at result SPU byte t (selector 0x10+t), and the
 * insert positions select a's preferred scalar bytes -- SPU byte 3 for a byte
 * (selector 0x03), SPU bytes 2,3 for a halfword (0x02,0x03), SPU bytes 0..3 for
 * a word, 0..7 for a doubleword. Every store is mapped through SPU_W to land at
 * the right host index. Verified to compose correctly with the fixed shufb for
 * both cbd-generated and immediate/LS-constant controls. */
static inline u128 spu_cbd_pos_ref(int pos) {
    u128 r; for(int t=0;t<16;t++) r._u8[SPU_W(t)] = (uint8_t)(0x10 + t);
    r._u8[SPU_W(pos & 0xF)] = 0x03;                 /* a's preferred byte = SPU byte 3 */
    return r;
}
#if SPU__SIMD
static inline u128 spu_cbd_pos(int pos){ return spu__ldtbl(spu__t_cbd[pos&0xF]); }
#else
#define spu_cbd_pos spu_cbd_pos_ref
#endif
static inline u128 spu_chd_pos_ref(int pos) {
    u128 r; for(int t=0;t<16;t++) r._u8[SPU_W(t)] = (uint8_t)(0x10 + t);
    int p = (pos & 0xF) & ~1;                       /* SPU halfword byte positions p, p+1 */
    r._u8[SPU_W(p)]   = 0x02;                        /* a SPU byte 2 (hi byte of preferred hw) */
    r._u8[SPU_W(p+1)] = 0x03;                        /* a SPU byte 3 (lo byte) */
    return r;
}
#if SPU__SIMD
static inline u128 spu_chd_pos(int pos){ return spu__ldtbl(spu__t_chd[pos&0xF]); }
#else
#define spu_chd_pos spu_chd_pos_ref
#endif
static inline u128 spu_cwd_pos_ref(int pos) {
    u128 r; for(int t=0;t<16;t++) r._u8[SPU_W(t)] = (uint8_t)(0x10 + t);
    int p = (pos & 0xF) & ~3;
    for(int k=0;k<4;k++) r._u8[SPU_W(p+k)] = (uint8_t)k;   /* a SPU bytes 0..3 (preferred word) */
    return r;
}
#if SPU__SIMD
static inline u128 spu_cwd_pos(int pos){ return spu__ldtbl(spu__t_cwd[pos&0xF]); }
#else
#define spu_cwd_pos spu_cwd_pos_ref
#endif
static inline u128 spu_cdd_pos_ref(int pos) {
    u128 r; for(int t=0;t<16;t++) r._u8[SPU_W(t)] = (uint8_t)(0x10 + t);
    int p = (pos & 0xF) & ~7;
    for(int k=0;k<8;k++) r._u8[SPU_W(p+k)] = (uint8_t)k;   /* a SPU bytes 0..7 (preferred dword) */
    return r;
}
#if SPU__SIMD
static inline u128 spu_cdd_pos(int pos){ return spu__ldtbl(spu__t_cdd[pos&0xF]); }
#else
#define spu_cdd_pos spu_cdd_pos_ref
#endif
static inline u128 spu_cbd(u128 a, int i7){ return spu_cbd_pos((int)a._u32[0]+i7); }
static inline u128 spu_chd(u128 a, int i7){ return spu_chd_pos((int)a._u32[0]+i7); }
static inline u128 spu_cwd(u128 a, int i7){ return spu_cwd_pos((int)a._u32[0]+i7); }
static inline u128 spu_cdd(u128 a, int i7){ return spu_cdd_pos((int)a._u32[0]+i7); }
static inline u128 spu_cbx(u128 a, u128 b){ return spu_cbd_pos((int)(a._u32[0]+b._u32[0])); }
static inline u128 spu_chx(u128 a, u128 b){ return spu_chd_pos((int)(a._u32[0]+b._u32[0])); }
static inline u128 spu_cwx(u128 a, u128 b){ return spu_cwd_pos((int)(a._u32[0]+b._u32[0])); }
static inline u128 spu_cdx(u128 a, u128 b){ return spu_cdd_pos((int)(a._u32[0]+b._u32[0])); }

/* ---- Phase 3: rotate-and-mask family ---- */
static inline u128 spu_rotm(u128 a, u128 b)   { u128 r; for(int i=0;i<4;i++){ uint32_t sh=(0-b._u32[i])&0x3F; r._u32[i]=(sh>31)?0:(a._u32[i]>>sh); } return r; }
static inline u128 spu_rotma(u128 a, u128 b)  { u128 r; for(int i=0;i<4;i++){ uint32_t sh=(0-b._u32[i])&0x3F; r._s32[i]=(sh>31)?(a._s32[i]>>31):(a._s32[i]>>sh); } return r; }
static inline u128 spu_rothm(u128 a, u128 b)  { u128 r; for(int i=0;i<8;i++){ uint32_t sh=(0-b._u16[i])&0x1F; r._u16[i]=(sh>15)?0:(uint16_t)(a._u16[i]>>sh); } return r; }
static inline u128 spu_rothma(u128 a, u128 b) { u128 r; for(int i=0;i<8;i++){ uint32_t sh=(0-b._u16[i])&0x1F; r._s16[i]=(sh>15)?(a._s16[i]>>15):(a._s16[i]>>sh); } return r; }
static inline u128 spu_rothmi(u128 a, int i7) { u128 r; int sh=(0-i7)&0x1F; for(int i=0;i<8;i++) r._u16[i]=(sh>15)?0:(uint16_t)(a._u16[i]>>sh); return r; }
static inline u128 spu_rotqmbi_ref(u128 a, u128 b)   { int sh=(0-(int)b._u32[0])&7; if(!sh) return a;
    uint64_t hi=((uint64_t)a._u32[0]<<32)|a._u32[1], lo=((uint64_t)a._u32[2]<<32)|a._u32[3];
    uint64_t nlo=(lo>>sh)|(hi<<(64-sh)), nhi=(hi>>sh);
    u128 r; r._u32[0]=(uint32_t)(nhi>>32); r._u32[1]=(uint32_t)nhi; r._u32[2]=(uint32_t)(nlo>>32); r._u32[3]=(uint32_t)nlo; return r; }
static inline u128 spu_rotqmby_ref(u128 a, u128 b)   { int sh=(0-(int)b._u32[0])&0x1F; u128 r=spu_zero(); if(sh>=16) return r; for(int i=0;i<16;i++){ int s=i-sh; if(s>=0) r._u8[SPU_W(i)]=a._u8[SPU_W(s)]; } return r; }
#if SPU__SIMD
static inline u128 spu_rotqmby(u128 a, u128 b){ int sh=(0-(int)b._u32[0])&0x1F; return sh>=16 ? spu_zero() : spu__pshufb(a, spu__t_shr[sh]); }
#else
#define spu_rotqmby spu_rotqmby_ref
#endif
static inline u128 spu_rotqmbybi_ref(u128 a, u128 b) { int sh=(0-((int)b._u32[0]>>3))&0x1F; u128 r=spu_zero(); if(sh>=16) return r; for(int i=0;i<16;i++){ int s=i-sh; if(s>=0) r._u8[SPU_W(i)]=a._u8[SPU_W(s)]; } return r; }
#if SPU__SIMD
static inline u128 spu_rotqmbybi(u128 a, u128 b){ int sh=(0-((int)b._u32[0]>>3))&0x1F; return sh>=16 ? spu_zero() : spu__pshufb(a, spu__t_shr[sh]); }
#else
#define spu_rotqmbybi spu_rotqmbybi_ref
#endif
static inline u128 spu_rotqmbii_ref(u128 a, int i7)  { int sh=(0-i7)&7; if(!sh) return a;
    uint64_t hi=((uint64_t)a._u32[0]<<32)|a._u32[1], lo=((uint64_t)a._u32[2]<<32)|a._u32[3];
    uint64_t nlo=(lo>>sh)|(hi<<(64-sh)), nhi=(hi>>sh);
    u128 r; r._u32[0]=(uint32_t)(nhi>>32); r._u32[1]=(uint32_t)nhi; r._u32[2]=(uint32_t)(nlo>>32); r._u32[3]=(uint32_t)nlo; return r; }
static inline u128 spu_rotqmbyi_ref(u128 a, int i7)  { int sh=(0-i7)&0x1F; u128 r=spu_zero(); if(sh>=16) return r; for(int i=0;i<16;i++){ int s=i-sh; if(s>=0) r._u8[SPU_W(i)]=a._u8[SPU_W(s)]; } return r; }
#if SPU__SIMD
static inline u128 spu_rotqmbyi(u128 a, int i7){ int sh=(0-i7)&0x1F; return sh>=16 ? spu_zero() : spu__pshufb(a, spu__t_shr[sh]); }
#else
#define spu_rotqmbyi spu_rotqmbyi_ref
#endif

/* ---- Phase 3: halfword/byte immediate logic ---- */
static inline u128 spu_andhi(u128 a, int32_t imm) { u128 r; uint16_t v=(uint16_t)imm; for(int i=0;i<8;i++) r._u16[i]=a._u16[i]&v; return r; }
static inline u128 spu_andbi(u128 a, int32_t imm) { u128 r; uint8_t  v=(uint8_t)imm;  for(int i=0;i<16;i++) r._u8[i]=a._u8[i]&v;   return r; }
static inline u128 spu_orhi(u128 a, int32_t imm)  { u128 r; uint16_t v=(uint16_t)imm; for(int i=0;i<8;i++) r._u16[i]=a._u16[i]|v; return r; }
static inline u128 spu_orbi(u128 a, int32_t imm)  { u128 r; uint8_t  v=(uint8_t)imm;  for(int i=0;i<16;i++) r._u8[i]=a._u8[i]|v;   return r; }
static inline u128 spu_xorhi(u128 a, int32_t imm) { u128 r; uint16_t v=(uint16_t)imm; for(int i=0;i<8;i++) r._u16[i]=a._u16[i]^v; return r; }

/* ---- Phase 3: borrow-generate extended ---- */
static inline u128 spu_bgx(u128 a, u128 b, u128 t) {
    u128 r;
    for(int i=0;i<4;i++) {
        uint64_t s = (uint64_t)b._u32[i] + (uint64_t)(~a._u32[i]) + (uint64_t)(t._u32[i]&1);
        r._u32[i] = (uint32_t)(s >> 32);
    }
    return r;
}

/* ---- Phase 3: mfspr stub ---- */
static inline u128 spu_mfspr(u128 a) { (void)a; return spu_zero(); }

/* ---- immediate loaders ---- */
static inline u128 spu_il(int32_t imm)   { return spu_splat_u32((uint32_t)imm); }
static inline u128 spu_ila(uint32_t i18) { return spu_splat_u32(i18 & 0x3FFFF); }
static inline u128 spu_ilh(uint16_t imm) { return spu_splat_u16(imm); }
static inline u128 spu_ilhu(uint16_t imm){ return spu_splat_u32((uint32_t)imm << 16); }
static inline u128 spu_iohl(u128 a, uint16_t imm){ u128 r; for(int i=0;i<4;i++) r._u32[i]=a._u32[i]|(uint32_t)imm; return r; }

/* ---- NA1001 (perf, 2026-10-16): whole-register float compares / conversions / quadword bit shifts in EVERY
 * lifted TU, behind SPU_NA_FEAT bit 1 (0x2). The v130 forms in spu_helpers_vec.h only reached the register-cached
 * twins; the lifted job modules (inFamous 2's w2 batch jobs, spu/jobmod_*) include only this header, so fcgt was a
 * branchy 2x64-bit clamp + scalar compare chain, cflts/cfltu/cuflt scalar double loops, rotqbii 64-bit shift pairs.
 * Each form below is the same IEEE operation as its *_ref (same MXCSR rounding/FTZ/DAZ, the same spu__xfq clamp
 * that replaced spu_xfv); NaN lanes of a conversion use the reference. cfltu is new (no v130 form): lanes >= 2^31
 * are converted as (d - 2^31) ^ 0x80000000, exact because such a d is an integer multiple of 2^8. */
#if defined(__SSE4_1__) && SPU__FSIMD
#define SPU__NA2 (g_spu_na_feat & 2)
static SPU__AINLINE u128 spu_fcgt(u128 a, u128 b) {
    if (SPU__NA2) return spu__fst(_mm_cmpgt_ps(spu__xfq(spu__fld(a)), spu__xfq(spu__fld(b))));
    return spu_fcgt_ref(a, b);
}
static SPU__AINLINE u128 spu_fceq(u128 a, u128 b) {
    if (SPU__NA2) return spu__fst(_mm_cmpeq_ps(spu__xfq(spu__fld(a)), spu__xfq(spu__fld(b))));
    return spu_fceq_ref(a, b);
}
static SPU__AINLINE __m128 spu__na_fabs(__m128 v) { return _mm_and_ps(v, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF))); }
static SPU__AINLINE u128 spu_fcmgt(u128 a, u128 b) {
    if (SPU__NA2) return spu__fst(_mm_cmpgt_ps(spu__na_fabs(spu__xfq(spu__fld(a))), spu__na_fabs(spu__xfq(spu__fld(b)))));
    return spu_fcmgt_ref(a, b);
}
static SPU__AINLINE u128 spu_fcmeq(u128 a, u128 b) {
    if (SPU__NA2) return spu__fst(_mm_cmpeq_ps(spu__na_fabs(spu__xfq(spu__fld(a))), spu__na_fabs(spu__xfq(spu__fld(b)))));
    return spu_fcmeq_ref(a, b);
}
/* quadword BIT shifts/rotates by 0..7 (word 0 = most significant, as the references assemble it) */
static SPU__AINLINE __m128i spu__na_qshl(__m128i A, int sh, int rot) {
    const __m128i N = rot ? _mm_shuffle_epi32(A, _MM_SHUFFLE(0, 3, 2, 1)) : _mm_srli_si128(A, 4);   /* lane i <- lane i+1 */
    return _mm_or_si128(_mm_sll_epi32(A, _mm_cvtsi32_si128(sh)), _mm_srl_epi32(N, _mm_cvtsi32_si128(32 - sh)));
}
static SPU__AINLINE __m128i spu__na_qshr(__m128i A, int sh) {
    const __m128i P = _mm_slli_si128(A, 4);                                                          /* lane i <- lane i-1 */
    return _mm_or_si128(_mm_srl_epi32(A, _mm_cvtsi32_si128(sh)), _mm_sll_epi32(P, _mm_cvtsi32_si128(32 - sh)));
}
static SPU__AINLINE __m128i spu__na_vi(u128 a) { __m128i x; memcpy(&x, &a, 16); return x; }
static SPU__AINLINE u128 spu__na_vo(__m128i x) { u128 a; memcpy(&a, &x, 16); return a; }
static SPU__AINLINE u128 spu_shlqbii(u128 a, int sh)  { if (SPU__NA2) return spu__na_vo(spu__na_qshl(spu__na_vi(a), sh & 7, 0)); return spu_shlqbii_ref(a, sh); }
static SPU__AINLINE u128 spu_rotqbii(u128 a, int sh)  { if (SPU__NA2) return spu__na_vo(spu__na_qshl(spu__na_vi(a), sh & 7, 1)); return spu_rotqbii_ref(a, sh); }
static SPU__AINLINE u128 spu_shlqbi(u128 a, u128 b)   { if (SPU__NA2) return spu__na_vo(spu__na_qshl(spu__na_vi(a), (int)(b._u32[0] & 7), 0)); return spu_shlqbi_ref(a, b); }
static SPU__AINLINE u128 spu_rotqbi(u128 a, u128 b)   { if (SPU__NA2) return spu__na_vo(spu__na_qshl(spu__na_vi(a), (int)(b._u32[0] & 7), 1)); return spu_rotqbi_ref(a, b); }
static SPU__AINLINE u128 spu_rotqmbii(u128 a, int i7) { if (SPU__NA2) return spu__na_vo(spu__na_qshr(spu__na_vi(a), (0 - i7) & 7)); return spu_rotqmbii_ref(a, i7); }
static SPU__AINLINE u128 spu_rotqmbi(u128 a, u128 b)  { if (SPU__NA2) return spu__na_vo(spu__na_qshr(spu__na_vi(a), (0 - (int)b._u32[0]) & 7)); return spu_rotqmbi_ref(a, b); }
#if defined(__AVX2__)
/* ---- Q (perf, 2026-10-16): SPU_NA_FEAT bit 2 (0x4) -- float<->int conversions in single precision.
 * NA1001/v130 widen to __m256d (cvtps2pd + mul + min + max + cvttpd2dq, ~7 uops incl. two 256-bit lane
 * crossings) for every cflts/cfltu/csflt/cuflt. With the scale 2^k (k = 173-i8 or i8-155) a NORMAL float,
 * x * 2^k is exact whenever the result is normal, so the single-precision form is the same value:
 *  - cflts, k in [-126,126]: p = x*2^k (exact, or |p| < 1 when it would be subnormal -> both truncate to 0;
 *    or overflow -> +-inf/FLT_MAX >= 2^31 -> clamp, as the reference). cvttps2dq gives the indefinite
 *    0x80000000 for p >= 2^31; XOR with the (p >= 2^31) mask turns it into 0x7FFFFFFF. Denormal x: 0 both
 *    ways (|x*2^126| < 1). NaN lanes still go to the reference (caller checks).
 *  - cfltu: p = max(x*2^k, 0); [0,2^31) direct, [2^31,2^32) via (p-2^31) (exact: p is a multiple of 2^8),
 *    >= 2^32 -> 0xFFFFFFFF.
 *  - csflt, k in [-126,96] / cuflt, k in [-126,95]: round(int) * 2^k == round(int * 2^k) because the scaled
 *    result stays normal (|int| in [1,2^32]); the int->float rounding is ONE MXCSR rounding of the exact
 *    value, as the reference's double->float rounding. cuflt: hi16*65536 + lo16 = exact operands, one rounding.
 * Bit-identical to the references under every MXCSR mode (round, FTZ, DAZ); other k use the old path. */
#define SPU__Q_CVT 1
#define SPU__Q_ON (g_spu_na_feat & 4)
static SPU__AINLINE __m128 spu__q_p2f(int k) { return _mm_castsi128_ps(_mm_set1_epi32((127 + k) << 23)); }
static SPU__AINLINE __m128i spu__q_cflts(__m128 A, int k) {
    const __m128 p = k ? _mm_mul_ps(A, spu__q_p2f(k)) : A;
    return _mm_xor_si128(_mm_cvttps_epi32(p), _mm_castps_si128(_mm_cmpge_ps(p, _mm_set1_ps(2147483648.0f))));
}
static SPU__AINLINE __m128i spu__q_cfltu(__m128 A, int k) {
    __m128 p = k ? _mm_mul_ps(A, spu__q_p2f(k)) : A;
    p = _mm_max_ps(p, _mm_setzero_ps());
    const __m128i lo = _mm_cvttps_epi32(p);
    const __m128i hi = _mm_xor_si128(_mm_cvttps_epi32(_mm_sub_ps(p, _mm_set1_ps(2147483648.0f))), _mm_set1_epi32((int)0x80000000u));
    const __m128i r = _mm_blendv_epi8(lo, hi, _mm_castps_si128(_mm_cmpge_ps(p, _mm_set1_ps(2147483648.0f))));
    return _mm_or_si128(r, _mm_castps_si128(_mm_cmpge_ps(p, _mm_set1_ps(4294967296.0f))));
}
static SPU__AINLINE __m128 spu__q_csflt(__m128i I, int k) {
    const __m128 f = _mm_cvtepi32_ps(I);
    return k ? _mm_mul_ps(f, spu__q_p2f(k)) : f;
}
static SPU__AINLINE __m128 spu__q_cuflt(__m128i I, int k) {
    const __m128 hi = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srli_epi32(I, 16)), _mm_set1_ps(65536.0f));
    const __m128 f = _mm_add_ps(hi, _mm_cvtepi32_ps(_mm_and_si128(I, _mm_set1_epi32(0xFFFF))));
    return k ? _mm_mul_ps(f, spu__q_p2f(k)) : f;
}
static SPU__AINLINE u128 spu_cflts(u128 a, int i8) {
    if (SPU__Q_ON && 173 - i8 >= -126 && 173 - i8 <= 126) {
        const __m128 A = spu__xfq(spu__fld(a));
        if (!_mm_movemask_ps(_mm_cmpunord_ps(A, A))) return spu__na_vo(spu__q_cflts(A, 173 - i8));
    }
    if (SPU__NA2) {
        const __m128 A = spu__xfq(spu__fld(a));
        if (!_mm_movemask_ps(_mm_cmpunord_ps(A, A))) {
            __m256d d = _mm256_mul_pd(_mm256_cvtps_pd(A), _mm256_set1_pd(spu__p2(173 - i8)));
            d = _mm256_max_pd(_mm256_min_pd(d, _mm256_set1_pd(2147483647.0)), _mm256_set1_pd(-2147483648.0));
            return spu__na_vo(_mm256_cvttpd_epi32(d));
        }
    }
    return spu_cflts_ref(a, i8);
}
static SPU__AINLINE u128 spu_cfltu(u128 a, int i8) {
    if (SPU__Q_ON && 173 - i8 >= -126 && 173 - i8 <= 126) {
        const __m128 A = spu__xfq(spu__fld(a));
        if (!_mm_movemask_ps(_mm_cmpunord_ps(A, A))) return spu__na_vo(spu__q_cfltu(A, 173 - i8));
    }
    if (SPU__NA2) {
        const __m128 A = spu__xfq(spu__fld(a));
        if (!_mm_movemask_ps(_mm_cmpunord_ps(A, A))) {
            __m256d d = _mm256_mul_pd(_mm256_cvtps_pd(A), _mm256_set1_pd(spu__p2(173 - i8)));
            d = _mm256_min_pd(_mm256_max_pd(d, _mm256_setzero_pd()), _mm256_set1_pd(4294967295.0));
            /* d in [0, 2^31): exact truncation; d >= 2^31 gives the indefinite 0x80000000 there instead */
            const __m128i lo = _mm256_cvttpd_epi32(d);
            const __m128i hi = _mm_xor_si128(_mm256_cvttpd_epi32(_mm256_sub_pd(d, _mm256_set1_pd(2147483648.0))),
                                             _mm_set1_epi32((int)0x80000000u));
            return spu__na_vo(_mm_blendv_epi8(lo, hi, _mm_cmpeq_epi32(lo, _mm_set1_epi32((int)0x80000000u))));
        }
    }
    return spu_cfltu_ref(a, i8);
}
static SPU__AINLINE u128 spu_csflt(u128 a, int i8) {
    if (SPU__Q_ON && i8 - 155 >= -126 && i8 - 155 <= 96) return spu__fst(spu__q_csflt(spu__na_vi(a), i8 - 155));
    if (SPU__NA2) {
        const __m256d d = _mm256_mul_pd(_mm256_cvtepi32_pd(spu__na_vi(a)), _mm256_set1_pd(spu__p2(i8 - 155)));
        return spu__fst(_mm256_cvtpd_ps(d));
    }
    return spu_csflt_ref(a, i8);
}
static SPU__AINLINE u128 spu_cuflt(u128 a, int i8) {
    if (SPU__Q_ON && i8 - 155 >= -126 && i8 - 155 <= 95) return spu__fst(spu__q_cuflt(spu__na_vi(a), i8 - 155));
    if (SPU__NA2) {
        __m256d d = _mm256_cvtepi32_pd(spu__na_vi(a));
        d = _mm256_add_pd(d, _mm256_and_pd(_mm256_cmp_pd(d, _mm256_setzero_pd(), _CMP_LT_OQ), _mm256_set1_pd(4294967296.0)));
        return spu__fst(_mm256_cvtpd_ps(_mm256_mul_pd(d, _mm256_set1_pd(spu__p2(i8 - 155)))));
    }
    return spu_cuflt_ref(a, i8);
}
#else
static SPU__AINLINE u128 spu_cflts(u128 a, int i8) { return spu_cflts_ref(a, i8); }
static SPU__AINLINE u128 spu_cfltu(u128 a, int i8) { return spu_cfltu_ref(a, i8); }
static SPU__AINLINE u128 spu_csflt(u128 a, int i8) { return spu_csflt_ref(a, i8); }
static SPU__AINLINE u128 spu_cuflt(u128 a, int i8) { return spu_cuflt_ref(a, i8); }
#endif
#else
#define spu_fcgt     spu_fcgt_ref
#define spu_fceq     spu_fceq_ref
#define spu_fcmgt    spu_fcmgt_ref
#define spu_fcmeq    spu_fcmeq_ref
#define spu_shlqbii  spu_shlqbii_ref
#define spu_rotqbii  spu_rotqbii_ref
#define spu_shlqbi   spu_shlqbi_ref
#define spu_rotqbi   spu_rotqbi_ref
#define spu_rotqmbii spu_rotqmbii_ref
#define spu_rotqmbi  spu_rotqmbi_ref
#define spu_cflts    spu_cflts_ref
#define spu_cfltu    spu_cfltu_ref
#define spu_csflt    spu_csflt_ref
#define spu_cuflt    spu_cuflt_ref
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPU_HELPERS_H */
