/* spu_laneprof.h -- v96a [LANEPROF]: where SPURS lane time goes.
 *
 * Cheap per-thread accumulators (rdtsc) for the host time a SPURS policy-module
 * run (spu_run_policy_module, every lane of every workload) spends in runtime
 * helpers: MFC DMA, atomic line ops (GETLLAR/PUTLLC), other channel ops, the
 * SPU interpreter, spin backoff (yield/sleep) and the halt longjmp. The
 * remainder of the run is lifted SPU code + dispatch. spurs_policy.c folds each
 * run into per-workload totals and prints "[LANEPROF]" every 10 s.
 * LANEPROF=0 disables it (default on; it costs two rdtsc per helper call).
 * Must be included after spu_context.h (SPU_THREAD_LOCAL). */
#ifndef SPU_LANEPROF_H
#define SPU_LANEPROF_H
#include <stdint.h>
#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <x86intrin.h>
#endif

enum { LP_DMA = 0, LP_ATOM, LP_RDCH, LP_WRCH, LP_INTERP, LP_BACKOFF, LP_HALT, LP_NCAT };

typedef struct spu_laneprof_tls {
    uint64_t t[LP_NCAT];     /* rdtsc ticks per category (this run) */
    uint64_t n[LP_NCAT];     /* calls per category (this run)       */
    int      on;             /* inside a policy-module run          */
    int      depth;          /* nesting guard: outermost helper only */
    uint64_t halt_t0;        /* spu_halt -> setjmp return            */
    uint64_t llar, llar_rep; /* GETLLARs / ones that saw a line this thread already
                              * saw unchanged at its previous GETLLAR of it (polling) */
} spu_laneprof_tls;

extern SPU_THREAD_LOCAL spu_laneprof_tls g_spu_lp;

static inline uint64_t spu_lp_begin(void)
{
    if (!g_spu_lp.on) return 0;
    if (g_spu_lp.depth++) return 0;
    return __rdtsc();
}
static inline void spu_lp_end(uint64_t t0, int cat)
{
    if (!g_spu_lp.on) return;
    if (g_spu_lp.depth > 0) g_spu_lp.depth--;
    if (t0) { g_spu_lp.t[cat] += __rdtsc() - t0; g_spu_lp.n[cat]++; }
}
/* Time a region that cannot nest (backoff sleeps): always counted when on. */
static inline uint64_t spu_lp_raw_begin(void) { return g_spu_lp.on ? __rdtsc() : 0; }
static inline void spu_lp_raw_end(uint64_t t0, int cat)
{
    if (t0 && g_spu_lp.on) { g_spu_lp.t[cat] += __rdtsc() - t0; g_spu_lp.n[cat]++; }
}
#endif
