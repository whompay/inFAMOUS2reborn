/* spurs_policy.c — run a SPURS workload's policy module on a lifted SPU.
 *
 * The real SPURS kernel dispatches a workload by loading its policy module
 * (a raw SPU binary) at LS 0xA00 and entering it with the ABI of
 * cellSpursModuleEntry(uintptr_t context, uint64_t ea):
 *
 *   r0 = return-to-kernel address     r3 = kernel context LS address (0x100)
 *   r1 = stack (the PM re-loads its   r4 = the workload's u64 data (preferred
 *        own initial stack)                doubleword) — e.g. a joblist EA
 *                                     r5 = poll status
 *
 * The SpursKernelContext fields this fills, per the layout RPCS3 documents
 * (Emu/Cell/Modules/cellSpurs.h, struct SpursKernelContext) and the lifted
 * wwsjob PM verifiably reads. NOTE both pointers are 64-BIT:
 *   0x1C0 be u64 spurs EA           0x1D8 be u32 wklCurrentUniqueId
 *   0x1C8 be u32 spuNum             0x1DC be u32 wklCurrentId
 *   0x1CC be u32 dmaTagId           0x1E0 be u32 exitToKernelAddr
 *   0x1D0 be u64 wklCurrentAddr     0x1E4 be u32 selectWorkloadAddr
 *        (PM base 0xA00, in the LOW word at 0x1D4)
 *
 * exitToKernel/selectWorkload point at two reserved LS addresses below the PM
 * base; spu_indirect_branch intercepts them (ctx->policy_mode) and HLEs the
 * kernel side: select = "no contention, keep running", exit = end of run.
 */
#include "../../libs/video/rsx_getenv_cache.h"   /* cached getenv(): SPU hot paths called getenv per PM run (2026-10-04) */
#include "spu_context.h"
#include "spu_workload.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spu_laneprof.h"   /* v96a [LANEPROF] */
#include <time.h>

/* ---------------------------------------------------------------------------
 * v96a [LANEPROF] (LANEPROF=0 off): per-workload breakdown of SPURS lane time.
 * Every policy-module run (kernel-thread lane or lane-pool worker) is timed with
 * rdtsc; the runtime helpers it calls (spu_laneprof.h) are timed inside it. Each
 * thread folds its runs into thread-local per-wid sums and publishes them about
 * every 50 ms; the first thread past the 10 s mark prints one line per workload.
 * cellSpurs.c adds the lane-pool wake latency and the kernel's join wait.
 * -------------------------------------------------------------------------*/
enum { LPA_TOT = LP_NCAT, LPA_BUSYT, LPA_RUNS, LPA_BUSY, LPA_TW, LPA_LLAR, LPA_LLREP, LPA_N };
static volatile long long s_lpa[16][LPA_N];
static volatile long long s_lpa_n[16][LP_NCAT];
volatile long long g_lp_pool[16][4];   /* wake ticks, wakes, join-wait ticks, joins (cellSpurs.c) */
static SPU_THREAD_LOCAL long long t_lpa[16][LPA_N];
static SPU_THREAD_LOCAL long long t_lpa_n[16][LP_NCAT];
static SPU_THREAD_LOCAL unsigned long long t_lpa_next;
static SPU_THREAD_LOCAL int t_lpa_dirty;
static volatile unsigned long long s_lp_next_chk;
static volatile long s_lp_rep_busy;
static unsigned long long s_lp_win_tsc, s_lp_win_ns;
int spu_laneprof_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("LANEPROF"); on = (e && *e == '0') ? 0 : 1;
        fprintf(stderr, "[LANEPROF] v96a lane-time profiler %s (LANEPROF=0 off)\n", on ? "ON" : "off"); }
    return on;
}
static unsigned long long lp_now_ns(void)
{
    struct timespec ts; timespec_get(&ts, TIME_UTC);   /* C11; precise on UCRT and glibc */
    return (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
}
static void lp_flush(void)
{
    if (!t_lpa_dirty) return;
    for (int w = 0; w < 16; w++) {
        if (!t_lpa[w][LPA_RUNS]) continue;
        for (int c = 0; c < LPA_N; c++) if (t_lpa[w][c]) { __atomic_add_fetch(&s_lpa[w][c], t_lpa[w][c], __ATOMIC_RELAXED); t_lpa[w][c] = 0; }
        for (int c = 0; c < LP_NCAT; c++) if (t_lpa_n[w][c]) { __atomic_add_fetch(&s_lpa_n[w][c], t_lpa_n[w][c], __ATOMIC_RELAXED); t_lpa_n[w][c] = 0; }
    }
    t_lpa_dirty = 0;
}
static void lp_report(unsigned long long tsc)
{
    if (__atomic_exchange_n(&s_lp_rep_busy, 1, __ATOMIC_ACQUIRE)) return;
    unsigned long long ns = lp_now_ns();
    if (!s_lp_win_ns) { s_lp_win_ns = ns; s_lp_win_tsc = tsc; }
    else if (ns - s_lp_win_ns >= 10000000000ull) {
        const double win_s = (double)(ns - s_lp_win_ns) * 1e-9;
        const double tps = (double)(tsc - s_lp_win_tsc) / win_s;   /* rdtsc ticks per second */
        extern volatile int g_ab_mask;
        static const char* nm[LP_NCAT] = { "dma", "atom", "rdch", "wrch", "interp", "backoff", "halt" };
        for (int w = 0; w < 16; w++) {
            long long v[LPA_N], n[LP_NCAT];
            for (int c = 0; c < LPA_N; c++) v[c] = __atomic_exchange_n(&s_lpa[w][c], 0, __ATOMIC_RELAXED);
            for (int c = 0; c < LP_NCAT; c++) n[c] = __atomic_exchange_n(&s_lpa_n[w][c], 0, __ATOMIC_RELAXED);
            long long pw = __atomic_exchange_n(&g_lp_pool[w][0], 0, __ATOMIC_RELAXED), pwn = __atomic_exchange_n(&g_lp_pool[w][1], 0, __ATOMIC_RELAXED);
            long long pj = __atomic_exchange_n(&g_lp_pool[w][2], 0, __ATOMIC_RELAXED), pjn = __atomic_exchange_n(&g_lp_pool[w][3], 0, __ATOMIC_RELAXED);
            if (!v[LPA_RUNS] || v[LPA_TOT] <= 0) continue;
            const double tot = (double)v[LPA_TOT];
            double helpers = 0;
            char buf[1024]; int p = 0;
            p += snprintf(buf + p, sizeof buf - p, "[LANEPROF] w%d: %lld runs (%lld busy), lane time %.0f ms (busy runs %.0f ms, %.0f us/busy run) |",
                          w, v[LPA_RUNS], v[LPA_BUSY], 1000.0 * tot / tps, 1000.0 * v[LPA_BUSYT] / tps,
                          v[LPA_BUSY] ? 1e6 * v[LPA_BUSYT] / tps / (double)v[LPA_BUSY] : 0.0);
            for (int c = 0; c < LP_NCAT; c++) {
                if (c != LP_BACKOFF) helpers += (double)v[c];   /* backoff runs inside atom */
                p += snprintf(buf + p, sizeof buf - p, " %s %.1f%%/%lldk", nm[c], 100.0 * v[c] / tot, n[c] / 1000);
            }
            p += snprintf(buf + p, sizeof buf - p, " | lifted+dispatch %.1f%% | during PPU taskwait %.0f ms | getllar %lldk (%.0f%% re-polls of an unchanged line)",
                          100.0 * (tot - helpers) / tot, 1000.0 * v[LPA_TW] / tps, v[LPA_LLAR] / 1000,
                          v[LPA_LLAR] ? 100.0 * v[LPA_LLREP] / (double)v[LPA_LLAR] : 0.0);
            if (pwn) p += snprintf(buf + p, sizeof buf - p, " | pool wake %.1f us x%lld", 1e6 * pw / tps / (double)pwn, pwn);
            if (pjn) p += snprintf(buf + p, sizeof buf - p, ", join wait %.1f us x%lld", 1e6 * pj / tps / (double)pjn, pjn);
            fprintf(stderr, "%s (ab %d)\n", buf, g_ab_mask);
        }
        s_lp_win_ns = ns; s_lp_win_tsc = tsc;
    }
    __atomic_store_n(&s_lp_rep_busy, 0, __ATOMIC_RELEASE);
}
/* v98f: the per-helper timing (two rdtsc around every channel op / DMA / atomic / interpreter
 * call, plus the GETLLAR re-poll compare) cost ~8% of a w6/w10 lane run at 30-50M helper calls per
 * 10 s. Run totals (runs, busy, lane time, taskwait share) stay exact on every run; the helper
 * breakdown is taken on 1 run in LANEPROF_SAMPLE (default 8, per thread) and scaled by that factor.
 * LANEPROF_SAMPLE=1 = every run (v96a behaviour, also forced in PS3_AB phase 1). The re-poll %
 * then compares against the lane's previous SAMPLED GETLLAR of a line. */
static SPU_THREAD_LOCAL unsigned t_lp_ctr;
static SPU_THREAD_LOCAL long long t_lp_w;   /* weight of the current run's helper sums (0 = not sampled) */
static int spu_lp_detail_this_run(void)
{
    static int k = -1;
    if (k < 0) { const char* e = getenv("LANEPROF_SAMPLE"); int v = e ? atoi(e) : 8; if (v < 1) v = 1; if (v > 1024) v = 1024;
        k = v; fprintf(stderr, "[LANEPROF] v98f helper timing on 1 of %d runs per thread (LANEPROF_SAMPLE; PS3_AB phase 1 = every run)\n", k); }
    extern volatile int g_ab_mask;
    int kk = (g_ab_mask & 1) ? 1 : k;
    if (kk == 1) return 1;
    if ((++t_lp_ctr % (unsigned)kk) == 0) return kk;
    return 0;
}
/* Fold one finished policy-module run (ticks t0..now) into this thread's sums. */
static void lp_run_done(uint32_t wid, unsigned long long t0, int busy, int tw)
{
    unsigned long long now = __rdtsc(), d = now - t0;
    long long* a = t_lpa[wid & 15];
    a[LPA_TOT] += (long long)d; a[LPA_RUNS]++;
    if (tw) a[LPA_TW] += (long long)d;
    const long long w = t_lp_w;   /* v98f sampling weight */
    a[LPA_LLAR] += (long long)g_spu_lp.llar * w; a[LPA_LLREP] += (long long)g_spu_lp.llar_rep * w;
    if (busy) { a[LPA_BUSYT] += (long long)d; a[LPA_BUSY]++; }
    if (w) for (int c = 0; c < LP_NCAT; c++) { a[c] += (long long)g_spu_lp.t[c] * w; t_lpa_n[wid & 15][c] += (long long)g_spu_lp.n[c] * w; }
    t_lpa_dirty = 1;
    if (now >= t_lpa_next) {
        t_lpa_next = now + 100000000ull;   /* ~25-50 ms at 2-4 GHz */
        lp_flush();
        if (now >= s_lp_next_chk) { s_lp_next_chk = now + 400000000ull; lp_report(now); }
    }
}

extern int spu_run_with_halt(void (*)(spu_context*), spu_context*);

/* Counters the intercepts in spu_channels.c maintain for the CURRENT run
 * (single writer per run; reads are diagnostic). */
/* Thread-local since 2026-10-04: several SPURS kernel threads per instance
 * (SPURS_WKL_THREADS) run policy modules concurrently; each reads its own run's
 * counts right after the run. */
#ifndef SPU_PM_TLS
#ifdef _MSC_VER
#define SPU_PM_TLS __declspec(thread)
#else
#define SPU_PM_TLS _Thread_local
#endif
#endif
SPU_PM_TLS volatile unsigned g_spurs_pm_polls = 0;   /* selectWorkload calls this run */
SPU_PM_TLS volatile unsigned g_spurs_pm_exited = 0;  /* set when exitToKernel was taken */
/* v31: one-shot hook called on the FIRST selectWorkload poll of a run (= the module
 * found work). cellSpurs.c arms it for rotate-mode runs of SPURS_PAR_LANES workloads so
 * the other lanes start helping while this lane drains the batch. */
SPU_PM_TLS void (*g_spurs_pm_poll_hook)(void) = 0;
volatile unsigned g_wws_batch_gets = 0;   /* loadCommands (LS 0xC00) fetch count, spu_dma.h */

/* v31d stall breaker: per (wid, lane) "start fresh on the next run" requests from the
 * SPURS kernel (cellSpurs.c). A fresh context = the same reset a workload-data change
 * does (zeroed LS, PM image reloaded, full attach->claim cycle; SPURS_PM_FRESH=1 ran
 * every dispatch that way). */
static volatile unsigned char s_pm_force_fresh[16][6];
void spurs_pm_force_fresh(uint32_t wid)
{
    if (wid < 16) for (int i = 0; i < 6; i++) s_pm_force_fresh[wid][i] = 1;
}

static spu_context* volatile s_pm_ctx_pub[16][6];   /* v31h: mirror of s_pm_ctx for the hang dump */
static volatile uint64_t s_pm_data_pub[16][6];
/* v31h: hang-dump view of every lane's persistent PM context of one workload. */
void spurs_pm_dump_lanes(uint32_t wid)
{
    if (wid >= 16) return;
    for (int l = 0; l < 6; l++) {
        spu_context* c = s_pm_ctx_pub[wid][l];
        if (!c) continue;
        fprintf(stderr, "[lanes] w%u lane %d ctx=%p data=0x%llX pc=0x%05X status=0x%X\n", wid, l, (void*)c,
                (unsigned long long)s_pm_data_pub[wid][l], (unsigned)c->pc, (unsigned)c->status);
        for (uint32_t o = 0x1260; o < 0x1300; o += 32) {
            fprintf(stderr, "[lanes]   LS %05X:", o);
            for (int w = 0; w < 8; w++) {
                const uint8_t* p = c->ls + o + w * 4;
                fprintf(stderr, " %02X%02X%02X%02X", p[0], p[1], p[2], p[3]);
            }
            fprintf(stderr, "\n");
        }
        { const uint8_t* p = c->ls + 0x14B0;
          fprintf(stderr, "[lanes]   LS 014B0:");
          for (int w = 0; w < 8; w++) fprintf(stderr, " %02X%02X%02X%02X", p[w*4], p[w*4+1], p[w*4+2], p[w*4+3]);
          fprintf(stderr, "\n"); }
    }
}

/* v103B [FREEZE-SPURS]: one compact entry per lane of a workload for the F8 dump: pc, status, the data the
 * context was last run with, and the WWS job pipeline words (LS 0x12A0 loadJobState / 0x12C0 nextLoad /
 * 0x12D0 load / 0x12E0 run / 0x12F0 store, first word of each quad) + the cached queue line head (0x14B0). */
void spurs_pm_lane_brief(uint32_t wid, char* b, int cap)
{
    int p = 0;
    if (!b || cap < 2) return;
    b[0] = 0;
    if (wid >= 16) return;
    for (int l = 0; l < 6 && p < cap - 160; l++) {
        spu_context* c = s_pm_ctx_pub[wid][l];
        if (!c) continue;
        const uint8_t* ls = *(const uint8_t* volatile*)&c->ls;
        if (!ls) continue;   /* lane being reset right now (memset + init) */
        #define LW(o) ((unsigned)ls[(o)] << 24 | (unsigned)ls[(o)+1] << 16 | (unsigned)ls[(o)+2] << 8 | (unsigned)ls[(o)+3])
        p += snprintf(b + p, (size_t)(cap - p), " | l%d pc %05X st %X data %llX ff %u pipe %08X %08X %08X %08X %08X q %08X %08X",
                      l, (unsigned)c->pc, (unsigned)c->status, (unsigned long long)s_pm_data_pub[wid][l],
                      (unsigned)s_pm_force_fresh[wid][l], LW(0x12A0), LW(0x12C0), LW(0x12D0), LW(0x12E0), LW(0x12F0), LW(0x14B0), LW(0x14B4));
        #undef LW
    }
}

int spu_run_policy_module(spu_lifted_entry_fn entry, int image_id,
                          const uint8_t* pm_host, uint32_t pm_size,
                          uint64_t wkl_data, uint32_t wid, uint32_t spurs_ea,
                          uint32_t spu_num)
{
    if (!entry || !pm_host || !pm_size || pm_size > SPU_LS_SIZE - 0xA00)
        return -1;

    /* Reuse one context per thread: this runs per policy dispatch (thousands
     * per second with the SPURS workload churn live), and a fresh 260KB
     * calloc/free each time was allocator + page-zeroing churn. The explicit
     * memset preserves the exact fresh-zero semantics calloc provided. */
    /* PERSISTENT context per (workload, virtual SPU): a real policy module
     * stays RESIDENT on its SPU -- its LS carries continuation state between
     * scheduling quanta (the wwsjob PM caches the last-seen queue line at LS
     * 0x14B0 and its staging rings at 0x12xx; a freshly zeroed context every
     * dispatch made it treat every re-entry as first contact and it could
     * never progress its claim/stage pipeline). Image + context init happen
     * once; re-entries keep LS and only refresh the entry registers and the
     * (possibly game-updated) kernel-context fields. */
    enum { SPURS_PM_MAX_WKL = 16 };
    static spu_context* s_pm_ctx[SPURS_PM_MAX_WKL][6];
    if (wid >= SPURS_PM_MAX_WKL || spu_num >= 6) return -1;
    spu_context* ctx = s_pm_ctx[wid][spu_num];
    int first = 0;
    /* SPURS_PM_FRESH=1: treat EVERY dispatch as first contact (zeroed LS, image
     * reloaded). A/B against the persistent default: with persistence, the wwsjob
     * PM's first entry attaches + claims exactly ONE ticket then yields, and every
     * RE-entry takes a stale-state path that does NOTHING (LS 0x12A0/0x12C0/0x12F0
     * job-pipeline slots persist; entry only partially re-inits them) -- LBP's
     * 10-ticket loading queue stalls at lanes {1,2} forever. Fresh contexts make
     * each dispatch run the full attach->claim->yield cycle, so repeated kernel
     * dispatches drain the queue one ticket at a time (the persistent-LS claim
     * that motivated the cache predates the r4 word0/word1 layout fix). */
    static int s_fresh = -1;
    if (s_fresh < 0) s_fresh = getenv("SPURS_PM_FRESH") ? 1 : 0;
    /* Reset the persistent context when the workload's DATA (queue EA)
     * changes: the game re-points a wid at a NEW job queue once the old one
     * drains, and a real SPURS kernel reloads the policy module fresh on a
     * workload switch. Carrying queue A's LS (header cache 0x14B0, busy
     * masks 0x1270/0x1280, records) into queue B made the PM see stale
     * state and claim nothing (observed: LBP's second loading queue sat at
     * lanes {0,...} while the first drained fine). */
    static uint64_t s_pm_data[SPURS_PM_MAX_WKL][6];
    if (!ctx) {
        ctx = (spu_context*)malloc(sizeof(spu_context));
        if (!ctx) return -1;
        s_pm_ctx[wid][spu_num] = ctx;
        first = 1;
        s_pm_force_fresh[wid][spu_num] = 0;   /* v102: a brand-new context is fresh; a pending request (AddWorkload) must not reset it again on the next run */
        /* SPU_CTX_LOG=1: policy-module contexts are PERSISTENT and there is one
         * per (workload, spu_num) -- a workload with readyCount 8 legitimately
         * runs on several simulated SPUs, which then contend its lock lines
         * exactly as real SPUs would. Counting only the job contexts in
         * spu_lifted_job.h reports "two contexts" for a run that has five, and
         * makes correct multi-SPU execution look like duplicated state. */
        { static int s_cl = -1;
          if (s_cl < 0) s_cl = getenv("SPU_CTX_LOG") ? 1 : 0;
          if (s_cl) fprintf(stderr, "[spu-ctx] new POLICY context %p wid=%u spu_num=%u\n",
                            (void*)ctx, wid, spu_num); }
        memset(ctx, 0, sizeof(*ctx));
        spu_context_init(ctx, 0);
    } else if (s_pm_force_fresh[wid][spu_num]) {
        s_pm_force_fresh[wid][spu_num] = 0;
        first = 1;
        memset(ctx, 0, sizeof(*ctx));
        spu_context_init(ctx, 0);
    } else if (s_fresh || s_pm_data[wid][spu_num] != wkl_data) {
        if (!s_fresh) {
            static int _n = 0;
            if (_n++ < 400)   /* v31c: was 8 -- need every list switch near a freeze */
                fprintf(stderr, "[spurs-pm] wid=%u spu=%u workload data changed "
                        "0x%llX -> 0x%llX: fresh PM context\n", wid, spu_num,
                        (unsigned long long)s_pm_data[wid][spu_num],
                        (unsigned long long)wkl_data);
        }
        first = 1;
        memset(ctx, 0, sizeof(*ctx));
        spu_context_init(ctx, 0);
    }
    s_pm_data[wid][spu_num] = wkl_data;
    s_pm_ctx_pub[wid][spu_num] = ctx; s_pm_data_pub[wid][spu_num] = wkl_data;
    ctx->image_id    = image_id;
    ctx->policy_mode = 1;

    /* PM image at its load base (first entry only -- LS persists). */
    if (first)
        memcpy(ctx->ls + 0xA00, pm_host, pm_size);

    /* SpursKernelContext at LS 0x100 (fields big-endian, like all guest data). */
    uint8_t* ls = ctx->ls;
#define KBE32(off, v) do { uint32_t _v = (v); ls[(off)] = (uint8_t)(_v >> 24); \
        ls[(off)+1] = (uint8_t)(_v >> 16); ls[(off)+2] = (uint8_t)(_v >> 8);   \
        ls[(off)+3] = (uint8_t)_v; } while (0)
    /* wklCurrentAddr is a 64-BIT pointer (vm::bcptr<void,u64>), so it occupies
     * 0x1D0..0x1D7 and everything after it sits 4 bytes higher than a u32 would
     * put it. Writing it as a u32 shifted the whole tail of the struct:
     * exitToKernelAddr landed on selectWorkloadAddr's slot and selectWorkloadAddr
     * fell off into moduleId, leaving 0x1E4 zero. The wwsjob PM reads both --
     * `lqa $2,0x1e0; bisl $0,$2` at LS 0x3588 calls exitToKernel, and
     * `lqa $7,0x1e0; rotqbyi $6,$7,4; bisl $0,$6` at LS 0x35A0 calls
     * selectWorkload -- so it called our exit intercept for every poll and
     * branched to 0 for every select. It also tests wklCurrentId (0x1DC) against
     * 1, which only reads sensibly at the correct offset. */
    KBE32(0x1C0, 0);                    /* spurs EA hi32 (u64 ptr)  */
    KBE32(0x1C4, spurs_ea);             /* spurs EA lo32            */
    KBE32(0x1C8, spu_num);              /* spuNum (virtual SPU / lane row) */
    KBE32(0x1CC, 8);                    /* dmaTagId (kernel's tag)  */
    KBE32(0x1D0, 0);                    /* wklCurrentAddr hi32 (u64 ptr) */
    KBE32(0x1D4, 0xA00);                /* wklCurrentAddr lo32 = PM load base */
    KBE32(0x1D8, wid);                  /* wklCurrentUniqueId       */
    KBE32(0x1DC, wid);                  /* wklCurrentId             */
    KBE32(0x1E0, SPURS_PM_EXIT_TO_KERNEL_LS);
    KBE32(0x1E4, SPURS_PM_SELECT_WORKLOAD_LS);
#undef KBE32

    /* Entry registers per cellSpursModuleEntry. */
    ctx->gpr[0]._u32[0] = SPURS_PM_EXIT_TO_KERNEL_LS;  /* return-to-kernel link */
    ctx->gpr[1]._u32[0] = 0x3FFB0;                     /* stack (PM reloads its own) */
    /* ...except it does NOT: the PM's real entry is LS 0xA00, whose prologue
     * is `lqa $1,<stack slot>; ai $2,$1,32; il $5,0; stqd $2,0($1);
     * stqd $5,0x10($1); stqd $5,0x20($1); br <main>`. The lifted entry is
     * <main> (0x2BF0), so that prologue never ran and the PM -- and every job
     * it calls -- ran on 0x3FFB0, i.e. INSIDE the interrupt handler's register
     * save area (0x3FA00..0x3FFF0: stub at 0x3FEC0, r76-r79 at 0x3FE80). Each
     * stall-notify interrupt taken while a job ran overwrote the job's stack
     * frames (inFamous BatchJobList: garbage buffer pointers, output written
     * over its own command stream, `stop 0` trap). Emulate the prologue: find
     * `lqa $1,X` in the first 8 words of the PM and take its stack from LS X
     * (inFamous: 0x1540 -> 0x3F990). SPURS_PM_STACK=0 restores 0x3FFB0. */
    {   static int s_ps = -1;
        if (s_ps < 0) { const char* e = getenv("SPURS_PM_STACK"); s_ps = (e && *e == '0') ? 0 : 1; }
        for (uint32_t a = 0xA00; s_ps && a < 0xA20; a += 4) {
            uint32_t w = ((uint32_t)ls[a] << 24) | ((uint32_t)ls[a+1] << 16) | ((uint32_t)ls[a+2] << 8) | ls[a+3];
            if ((w >> 23) == 0x061 && (w & 0x7F) == 1) {          /* lqa $1, i16<<2 */
                uint32_t slot = (((w >> 7) & 0xFFFF) << 2) & 0x3FFF0;
                uint32_t sp = (((uint32_t)ls[slot] << 24) | ((uint32_t)ls[slot+1] << 16) |
                               ((uint32_t)ls[slot+2] << 8) | ls[slot+3]) & 0x3FFF0;
                if (sp >= 0x10000 && sp < 0x3FFF0) {
                    ctx->gpr[1] = spu_ls_read128(ctx, slot);   /* BE LS quad -> host register layout */
                    /* the backchain stores the prologue makes */
                    memset(ls + sp, 0, 48);
                    for (int k = 0; k < 4; k++) {               /* stqd ($1 + 32) */
                        uint32_t v = (((uint32_t)ls[slot+4*k] << 24) | ((uint32_t)ls[slot+4*k+1] << 16) |
                                      ((uint32_t)ls[slot+4*k+2] << 8) | ls[slot+4*k+3]) + 32u;
                        ls[sp+4*k] = (uint8_t)(v >> 24); ls[sp+4*k+1] = (uint8_t)(v >> 16);
                        ls[sp+4*k+2] = (uint8_t)(v >> 8); ls[sp+4*k+3] = (uint8_t)v;
                    }
                    static int _n = 0;
                    if (_n++ < 2) fprintf(stderr, "[spurs-pm] PM stack from LS 0x%05X: r1=0x%05X (reg %08X) (prologue at 0x%05X)\n", slot, sp, ctx->gpr[1]._u32[0], a);
                }
                break;
            }
        }
    }
    ctx->gpr[3]._u32[0] = 0x100;                       /* context */
    /* r4 layout is LO-WORD-IN-PREFERRED, not a big-endian u64: verified
     * against the authentic wwsjob PM bytes (entry 0x2BF0 `stqa r4,0x14E0`,
     * attach 0x2720 `brz r4`/`wrch MFC_EAL, r4` -- it treats r4's PREFERRED
     * WORD as the 32-bit queue EA, and `shlqbyi r4,4`'s next word as the aux
     * half). Passing {hi,lo} put 0 in the preferred slot, so the queue attach
     * early-returned on every run: tickets got claimed via the shifted copy
     * but the job-list header never DMA'd in, and the PM wedged forever in
     * its staged-job wait (LBP's post-intro loading freeze). */
    /* ...and the CLAIM path reads the EA from the shlqbyi(r4,4) copy (LS
     * 0x13F0, via func_3038 -> 2418), i.e. r4's SECOND word. Both word 0
     * (attach) and word 1 (claim) must carry the 32-bit EA; the hi half
     * rides in words 2-3. */
    /* 2026-10-04 (inFamous): the REAL kernel passes the 64-bit workload arg in
     * r4's preferred DOUBLEWORD (RPCS3 spursKernelDispatchWorkload:
     * gpr[4]._u64[1] = arg), i.e. word0 = arg>>32, word1 = arg & 0xFFFFFFFF.
     * The wwsjob PM's word-0 path (0x2C38 -> 0x2720) is an OPTIONAL per-SPU
     * audit/stat attach: it is skipped (`biz r4`) when the hi word is 0, which
     * it is for every 32-bit EA. With the EA in word 0 the PM treated the job
     * list header as an audit header and on EVERY PM entry PUT two quads to
     * hdr + 0x40 + 16*(spuNum + hdr[0x20]) -- for BatchJobList (hdr[0x20]=0xF)
     * that is 0x48C001B0..0x48C001FF, i.e. INSIDE job list 1. The PM then
     * walked list 1 past its end (row1 = 22) and list 2 never completed:
     * the ~30 s loading-screen/black-screen stall. The claim path reads
     * word 1 (shlqbyi r4,4 -> LS 0x13F0) and is unaffected.
     * SPURS_PM_R4_LEGACY=1 restores the old {lo,lo,hi,hi} layout (it was
     * introduced for LBP; re-test LBP with the real layout before relying on it). */
    static int s_r4_legacy = -1;
    if (s_r4_legacy < 0) { const char* e = getenv("SPURS_PM_R4_LEGACY"); s_r4_legacy = (e && *e == '1') ? 1 : 0;
        fprintf(stderr, "[spurs-pm] r4 layout: %s\n", s_r4_legacy ? "LEGACY {lo,lo,hi,hi}" : "real {hi,lo} (u64 in preferred dword)"); }
    if (s_r4_legacy) {
        ctx->gpr[4]._u32[0] = (uint32_t)wkl_data;          /* attach reads word0  */
        ctx->gpr[4]._u32[1] = (uint32_t)wkl_data;          /* claim reads word1   */
        ctx->gpr[4]._u32[2] = (uint32_t)(wkl_data >> 32);
        ctx->gpr[4]._u32[3] = (uint32_t)(wkl_data >> 32);
    } else {
        ctx->gpr[4]._u32[0] = (uint32_t)(wkl_data >> 32);  /* attach: 0 -> skipped */
        ctx->gpr[4]._u32[1] = (uint32_t)wkl_data;          /* claim reads word1   */
        ctx->gpr[4]._u32[2] = 0;
        ctx->gpr[4]._u32[3] = 0;
    }
    ctx->gpr[5]._u32[0] = 0;                           /* poll status */

    g_spurs_pm_polls  = 0;
    g_spurs_pm_exited = 0;

    /* RUN/END tracing: thousands of policy dispatches per second made these
     * unconditional fprintf+fflush pairs a real fraction of movie playback.
     * Default off; SPURS_PM_TRACE=1 restores them (first 64 + every 4096th
     * so an enabled trace can't flood either). */
    static int s_trace = -1;
    if (s_trace < 0) s_trace = getenv("SPURS_PM_TRACE") ? 1 : 0;
    static unsigned long s_runs = 0;
    unsigned long runno = s_trace ? ++s_runs : 0;
    int log_this = s_trace && (runno <= 64 || (runno & 0xFFF) == 0);
    if (log_this) {
        fprintf(stderr, "[spurs-pm] RUN#%lu wid=%u data=0x%llX spurs=0x%08X image=%d entry=0xA00 size=%u\n",
                runno, wid, (unsigned long long)wkl_data, spurs_ea, image_id, pm_size);
        fflush(stderr);
    }

    /* SPURS_PM_FLOW=1: capture this run's cross-function transfer trail (drain
     * hook in spu_drain.c) and dump it IF the run performed a PUTLLC (= a queue
     * claim) -- reconstructs the post-claim decision path that ends in a bare
     * exit-to-kernel instead of job staging. Only spu_num 0 is traced (the
     * concurrent lanes would interleave the buffer). */
    extern uint32_t g_pm_flow_buf[]; extern volatile unsigned g_pm_flow_n;
    extern void* volatile g_pm_flow_ctx;
    extern volatile unsigned g_spu_putllc_count;
    extern volatile unsigned g_spu_putllc_sync_hit;
    extern uint32_t g_barrier_sync_watch;
    static int s_flow = -1;
    if (s_flow < 0) s_flow = getenv("SPURS_PM_FLOW") ? 1 : 0;
    unsigned putllc_before = s_flow ? g_spu_putllc_sync_hit : 0;
    (void)g_spu_putllc_count;
    /* Trace ONLY the stalled queue's own runs: wkl_data == the sync EA the PPU
     * barrier/ticket probes armed (ticket-pub arms it at first publish, before
     * the PM's first work pass). Idle re-runs of this wid matter as much as the
     * claim run -- the empty-cursor path returns without any PUTLLC. */
    /* SPURS_PM_FLOW_ALL=1: trace EVERY spu_num-0 run of this workload without
     * requiring the PPU-side barrier probe to have armed g_barrier_sync_watch.
     * Needed to answer "does the job's buffer-binding routine (pm 0x3098, the
     * only writer of logicalToBufferNumArray@0x11B0 + jobState@0x12C0) ever
     * execute?" on a run that never reaches the barrier at all -- the RPCS3
     * guest dump shows a working SPU has 0x11B0 bound and runJobNum=0, ours
     * has 0x11B0 == 0 and runJobNum == -1. */
    static int s_flow_all = -1;
    if (s_flow_all < 0) s_flow_all = getenv("SPURS_PM_FLOW_ALL") ? 1 : 0;
    int tracing = (s_flow && spu_num == 0 && g_pm_flow_ctx == 0 &&
                   (s_flow_all ||
                    (g_barrier_sync_watch &&
                     (uint32_t)wkl_data == g_barrier_sync_watch)));
    static volatile unsigned long long s_flow_arm_ms;
    extern unsigned long long ps3_ms_now(void);
    if (tracing) {
        g_pm_flow_n = 0; g_pm_flow_ctx = ctx;
        s_flow_arm_ms = ps3_ms_now();
        { static int _a = 0; if (_a++ < 8) {
            fprintf(stderr, "[pm-flow] ARM wid=%u spu=%u (watch=0x%08X)\n",
                    wid, spu_num, g_barrier_sync_watch); fflush(stderr); } }
        /* Watchdog: the traced run has been observed to HANG the jobmanager
         * kernel thread (the claim run never returns -> no later dispatch can
         * report). A detached thread dumps the partial trail after 5s; the
         * trail's LAST pc names the looping lifted function. */
        { static int s_wd_started = 0;
          if (!s_wd_started) { s_wd_started = 1;
            extern void spurs_pm_flow_watchdog_start(void);
            spurs_pm_flow_watchdog_start(); } }
    }

    /* SPURS_PM_PIPE=1: trace how the WWS job PIPELINE state evolves across
     * kernel dispatches. Prints the pipeline counters (loadJobState@0x12A0
     * preferred hword, nextLoadJobNum@0x12C0, loadJobNum@0x12D0, runJobNum@
     * 0x12E0, storeJobNum@0x12F0) before and after each PM run, plus the pc
     * spu_run_with_halt halted at. Answers: does re-entry advance the pipeline
     * or reset/stall it? Only wid<=1 (jobmanager) traced, first 4000 runs. */
    static int s_pipe = -1;
    if (s_pipe < 0) s_pipe = getenv("SPURS_PM_PIPE") ? 1 : 0;
    /* trace ALL workloads (the jobmanager is wid=12, not <=1); 4000-run cap bounds volume */
    int pipe_this = s_pipe && spu_num == 0;
    uint32_t st_before = 0;
    if (pipe_this) {
        const uint8_t* L = ctx->ls;
        #define HW(o) ((L[(o)+2]<<8)|L[(o)+3])   /* preferred hword (bytes 2-3) */
        st_before = HW(0x12A0);
        (void)st_before;
        #undef HW
    }

    /* SPURS_PM_CLAIMLOG=1: per-run claim-vs-stage summary for BOTH lanes.
     * The loading freeze's signature is a dispatch that CLAIMS a ticket
     * (sync-PUTLLC hit) but never STAGES its job (no loadCommands GET): the
     * claimed ticket is eaten and the batch's barrier can never release
     * (observed: 4-job batch, counter=4, lanes {2,1}). Diffing the sync-PUTLLC
     * and 0xC00-GET counters across each run names that dispatch and shows the
     * LS pipeline state it left behind. */
    static int s_claimlog = -1;
    if (s_claimlog < 0) s_claimlog = getenv("SPURS_PM_CLAIMLOG") ? 1 : 0;
    unsigned batch_before = g_wws_batch_gets;
    unsigned claim_before = g_spu_putllc_sync_hit;

    /* WWS_LATE=<sec>[:<wid>]: after <sec> seconds, log the first 80 PM runs
     * (of <wid>, default any) with the job-list sync row 0..1 of the workload's
     * data before/after, the LS pipeline state and the stall/interrupt state.
     * For hangs that only start late (inFamous' BatchJobList, ~30 s in). */
    static long long s_late = -2; static int s_late_wid = -1; static unsigned long long s_late_t0;
    if (s_late == -2) { const char* e = getenv("WWS_LATE"); s_late = e ? atoll(e) : -1;
        if (e && strchr(e, ':')) s_late_wid = atoi(strchr(e, ':') + 1);
        extern unsigned long long ps3_ms_now(void); s_late_t0 = ps3_ms_now(); }
    int late_this = 0; uint8_t late_row_before[32];
    if (s_late >= 0 && (s_late_wid < 0 || (int)wid == s_late_wid)) {
        extern unsigned long long ps3_ms_now(void);
        static volatile long s_late_n;
        (void)s_late_n;
        if (ps3_ms_now() - s_late_t0 >= (unsigned long long)s_late * 1000ull) {
            late_this = 1;
            extern uint8_t* vm_base;
            memcpy(late_row_before, vm_base + (uint32_t)wkl_data + 0x40, 32);
        }
    }
    /* WWS_RUNWATCH=<ms>: a PM run that has not returned after <ms> is sampled
     * once a second by a watchdog thread (pc, link, depth, status, events,
     * interrupt/stall state, pipeline words) -- names the loop a stuck SPURS
     * kernel quantum is spinning in. */
    extern int spurs_runwatch_begin(void* ctx, unsigned wid, unsigned sn);
    extern void spurs_runwatch_end(void* ctx);
    int runwatched = spurs_runwatch_begin(ctx, wid, spu_num);
    { extern spu_lifted_entry_fn spu_rc_select(spu_lifted_entry_fn f);   /* v95: register-cached twin */
      entry = spu_rc_select(entry); }
    /* v96a [LANEPROF]: time this run and the helpers inside it (nest-safe). */
    spu_laneprof_tls lp_saved; unsigned long long lp_t0 = 0; int lp_w = 0;
    const int lp_on = spu_laneprof_on();
    extern volatile long g_spurs_ppu_tw;   /* cellSpurs.c: > 0 while the PPU spins in a CTaskMgr wait */
    const int lp_tw = lp_on && g_spurs_ppu_tw > 0;
    if (lp_on) { lp_saved = g_spu_lp; memset(&g_spu_lp, 0, sizeof g_spu_lp); lp_w = spu_lp_detail_this_run(); g_spu_lp.on = lp_w != 0; lp_t0 = __rdtsc(); }
    const int pm_halted = spu_run_with_halt(entry, ctx);
    if (lp_on) { g_spu_lp.on = 0; t_lp_w = lp_w; lp_run_done(wid, lp_t0, g_spurs_pm_polls != 0, lp_tw); g_spu_lp = lp_saved; }
    if (runwatched) spurs_runwatch_end(ctx);
    if (late_this) {
        extern uint8_t* vm_base;
        const uint8_t* a = vm_base + (uint32_t)wkl_data + 0x40; const uint8_t* b = late_row_before;
        const uint8_t* L = ctx->ls;
        /* Only log INTERESTING runs (row changed, row non-idle, pipeline busy,
         * claimed/staged something, or didn't exit to kernel) + a heartbeat
         * every 4096 idle runs; cap 300 lines. */
        static const uint8_t idle_row[16] = {0,0,0,0,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
        static volatile long s_late_idle, s_late_log;
        int interesting = memcmp(a, b, 32) != 0 || (first && memcmp(a, idle_row, 16) != 0) ||
            L[0x12E0] != 0xFF || L[0x12D0] != 0xFF || L[0x12F0] != 0xFF ||
            g_spu_putllc_sync_hit != claim_before || g_wws_batch_gets != batch_before ||
            ((uint32_t)ctx->pc & SPU_LS_MASK) != SPURS_PM_EXIT_TO_KERNEL_LS || ctx->list_stall_stat;
        if (!interesting && (__atomic_add_fetch(&s_late_idle, 1, __ATOMIC_RELAXED) & 4095) != 1) late_this = 0;
        if (late_this && __atomic_add_fetch(&s_late_log, 1, __ATOMIC_RELAXED) > 300) late_this = 0;
    }
    if (late_this) {
        extern uint8_t* vm_base;
        const uint8_t* a = vm_base + (uint32_t)wkl_data + 0x40; const uint8_t* b = late_row_before;
        const uint8_t* L = ctx->ls;
        #define W(p,o) (((uint32_t)(p)[(o)]<<24)|((uint32_t)(p)[(o)+1]<<16)|((uint32_t)(p)[(o)+2]<<8)|(p)[(o)+3])
        fprintf(stderr, "[wws-late] t=%llu wid=%u spu=%u first=%d halted=%d pc=0x%05X status=0x%X ev=0x%X/0x%X ie=%d "
                "stallmask=0x%X stallstat=0x%X | row0 %08X %08X %08X %08X -> %08X %08X %08X %08X"
                " | row1 %08X %08X -> %08X %08X | loadState=%08X nextLoad=%08X loadJob=%08X runJob=%08X storeJob=%08X\n",
                (unsigned long long)(ps3_ms_now() - s_late_t0), wid, spu_num, first, pm_halted, (uint32_t)ctx->pc & SPU_LS_MASK, ctx->status,
                ctx->event_status, ctx->event_mask, ctx->int_enable,
                ctx->list_stall_mask, ctx->list_stall_stat,
                W(b,0), W(b,4), W(b,8), W(b,12), W(a,0), W(a,4), W(a,8), W(a,12),
                W(b,16), W(b,20), W(a,16), W(a,20),
                W(L,0x12A0), W(L,0x12C0), W(L,0x12D0), W(L,0x12E0), W(L,0x12F0));
        #undef W
        fflush(stderr);
    }
    /* A lane's run should only end through exitToKernel. Anything else strands
     * the jobs it had claimed (GH3's once-per-~10-songs freeze): say how. */
    if (((uint32_t)ctx->pc & SPU_LS_MASK) != SPURS_PM_EXIT_TO_KERNEL_LS) {
        extern unsigned spu_recent_pcs(uint32_t*, unsigned);
        uint32_t r[32]; unsigned n = spu_recent_pcs(r, 32);
        static volatile long s_pe; if (__atomic_add_fetch(&s_pe, 1, __ATOMIC_RELAXED) <= 64) {
            fprintf(stderr, "[pm-end] lane %u ctx=%p halted=%d status=0x%X pc=0x%05X lr=0x%05X r3=0x%08X depth=%u recent:",
                    spu_num, (void*)ctx, pm_halted, ctx->status, (uint32_t)ctx->pc & SPU_LS_MASK,
                    ctx->gpr[0]._u32[0] & SPU_LS_MASK, ctx->gpr[3]._u32[0], ctx->host_depth);
            for (unsigned k = 0; k < n; k++) fprintf(stderr, " %05X", r[k]);
            fprintf(stderr, "\n"); fflush(stderr); } }

    if (s_claimlog) {
        unsigned claims = g_spu_putllc_sync_hit - claim_before;
        unsigned stages = g_wws_batch_gets - batch_before;
        if (claims || stages) {
            static unsigned s_cl = 0;
            if (s_cl++ < 64) {
                const uint8_t* L = ctx->ls;
                #define W(o)  ((L[(o)]<<24)|(L[(o)+1]<<16)|(L[(o)+2]<<8)|L[(o)+3])
                fprintf(stderr, "[pm-claim] wid=%u spu=%u first=%d claims=%u stages=%u%s halt_pc=0x%05X "
                        "nextLoad=0x%X loadJob=0x%X runJob=0x%X storeJob=0x%X q13F0=%08X q14E0=%08X\n",
                        wid, spu_num, first, claims, stages,
                        (claims && !stages) ? " <-- CLAIM EATEN" : "",
                        ctx->pc & SPU_LS_MASK,
                        W(0x12C0), W(0x12D0), W(0x12E0), W(0x12F0),
                        W(0x13F0), W(0x14E0));
                fflush(stderr);
                #undef W
            }
        }
    }

    if (pipe_this) {
        /* PER-WID cap: a shared cap gets exhausted by idle wid=0/1 before the
         * jobmanager (wid=12) ever runs in a long/good run. Give each wid its
         * own budget so wid=12's pipeline is always captured. */
        static unsigned long s_pn_wid[16] = {0};
        unsigned long s_pn = ++s_pn_wid[wid & 15];
        if (s_pn < 300) {
            const uint8_t* L = ctx->ls;
            #define HW(o) ((L[(o)+2]<<8)|L[(o)+3])
            #define W(o)  ((L[(o)]<<24)|(L[(o)+1]<<16)|(L[(o)+2]<<8)|L[(o)+3])
            fprintf(stderr, "[pm-pipe] wid=%u run#%lu first=%d halt_pc=0x%05X "
                    "loadState=%u->%u nextLoad=0x%X loadJob=0x%X runJob=0x%X storeJob=0x%X "
                    "lastStore=0x%X\n",
                    wid, s_pn, first, ctx->pc & SPU_LS_MASK,
                    st_before, HW(0x12A0), W(0x12C0), W(0x12D0), W(0x12E0),
                    W(0x12F0), W(0x1300));
            /* WHY-NO-JOB1 probe: AllocateJob (via TryChangeFreeToLoadJob) stores
             * its result at g_WwsJob_jobHeader(0x1420) + g_WwsJob_jobIndex(0x1410).
             * jobHeader low-hword: 0=none(PPU hasn't published)/1=exists/3=nop.
             * Also dump the joblist header in RAM (eaWorkLoad@0x13F0) to see the
             * PPU-published numJobs + claimed jobIndex -- decides whether the PM
             * stops because the header is unpublished (PPU-side) or because our
             * load path bails after claiming. */
            {
                extern uint8_t* vm_base;
                uint32_t eaWL = W(0x13F0);
                fprintf(stderr, "         jobIndex=0x%X jobHeader=0x%08X(cmd=%u) eaWL=0x%08X\n",
                        W(0x1410), W(0x1420), HW(0x1420), eaWL);
                /* Dump the joblist header region eaWL+0x00..0x9F so we can see the
                 * ticket/counter rows AND where the PPU actually wrote the job
                 * headers (ref AllocateJob reads eaWL+0x20; oracle layout puts
                 * jobHeaders at +0x80). If +0x20 is 0 but +0x80 has 0x00xx0001
                 * entries, our AllocateJob reads the WRONG offset. */
                if (vm_base && eaWL) {
                    for (uint32_t o = 0; o < 0xA0; o += 0x10) {
                        const uint8_t* r = vm_base + eaWL + o;
                        #define RW(i) ((r[i]<<24)|(r[i+1]<<16)|(r[i+2]<<8)|r[i+3])
                        fprintf(stderr, "         jl+%02X: %08X %08X %08X %08X\n",
                                o, RW(0), RW(4), RW(8), RW(12));
                        #undef RW
                    }
                }
            }
            fflush(stderr);
            #undef HW
            #undef W
        }
    }

    if (tracing) {
        g_pm_flow_ctx = 0;
        static int s_dumps = 0;
        /* Only dump once the PPU is actually spinning on a job barrier
         * (g_barrier_sync_watch armed by the barrier probe): boot-time idle
         * runs also PUTLLC (queue-check protocol) and would burn the cap. */
        /* Only dump runs that actually had a JOB STAGED. A boot-time idle run
         * has empty loadCommands(0xC00)/bufferSetArray(0xDF0) and its trail says
         * nothing about the Load->Run gate -- with SPURS_PM_FLOW_ALL tracing
         * every run, the dump budget was being spent entirely on idle runs. */
        int has_job = 0;
        for (uint32_t _i = 0xDF0; _i < 0xE10 && !has_job; _i++)
            if (ctx->ls[_i]) has_job = 1;
        for (uint32_t _i = 0xC00; _i < 0xC60 && !has_job; _i++)
            if (ctx->ls[_i]) has_job = 1;
        if (has_job && s_dumps < 6) {
            s_dumps++;
            unsigned n = g_pm_flow_n; if (n > 8192) n = 8192;
            fprintf(stderr, "[pm-flow] dump#%d wid=%u %u transfers (sync-PUTLLC delta %u):",
                    s_dumps, wid, n, g_spu_putllc_sync_hit - putllc_before);
            uint32_t last = 0xFFFFFFFFu; unsigned rep = 0;
            for (unsigned i = 0; i < n; i++) {
                uint32_t pc = g_pm_flow_buf[i] & SPU_LS_MASK;
                if (pc == last) { rep++; continue; }
                if (rep) { fprintf(stderr, "*%u", rep + 1); rep = 0; }
                fprintf(stderr, "%s%05X", (i % 16) ? " " : "\n  ", pc);
                last = pc;
            }
            if (rep) fprintf(stderr, "*%u", rep + 1);
            fprintf(stderr, "\n");
            /* Pipeline LS state right after this traced run: same regions the
             * hung-run watchdog dumps -- shows what the 5 type-1 passes built
             * (records/ring) and why the type-2 loads never dispatch. */
            { static const struct { uint32_t a, len; const char* tag; } R[] = {
                  { 0xC00, 0x60, "cmds" }, { 0xDF0, 0x40, "rec" },
                  { 0xEB0, 0x60, "ring" }, { 0x11B0, 0x20, "mark" },
                  { 0x1270, 0x20, "busy" }, { 0x12A0, 0x10, "st12A0" },
                  { 0x12C0, 0x40, "state" } };
              for (unsigned r = 0; r < sizeof R / sizeof R[0]; r++) {
                  fprintf(stderr, "[pm-ls] %-6s 0x%04X:", R[r].tag, R[r].a);
                  for (uint32_t o = 0; o < R[r].len; o += 4) {
                      if (o && (o & 15) == 0) fprintf(stderr, " |");
                      fprintf(stderr, " %02X%02X%02X%02X",
                              ctx->ls[R[r].a+o], ctx->ls[R[r].a+o+1],
                              ctx->ls[R[r].a+o+2], ctx->ls[R[r].a+o+3]);
                  }
                  fprintf(stderr, "\n");
              } }
            fflush(stderr);
        }
    }

    if (log_this) {
        fprintf(stderr, "[spurs-pm] END wid=%u status=0x%X pc=0x%05X polls=%u exited=%u\n",
                wid, ctx->status, ctx->pc, g_spurs_pm_polls, g_spurs_pm_exited);
        fflush(stderr);
    }

    return g_spurs_pm_exited ? 0 : -2;
}

/* ---------------------------------------------------------------------------
 * Real taskset-policy probe (Option 2, A/B vs spurs_pm_build_context).
 *
 * Runs the REAL lifted SPURS taskset policy (tsp_spu_func_00000A00, extracted
 * from libsre vaddr 0x23680) far enough to build the SpursTasksetContext at LS
 * 0x2700, then copies that region out for comparison against the C reimpl. The
 * policy's raw bytes (its data tables at LS 0x27B0-0x2840, read via lqr) come
 * from the embedded array so there is no dependency on libsre's guest load
 * address. The policy will eventually `bi` to the selected task's entry (an
 * address in the task image's LS range, unregistered under image id 100) and
 * halt as branch-to-0 -- by then LS 0x2700 holds the authentic context.
 *
 * image_id 100 is reserved for the policy's own indirect-branch table (the
 * lifted policy registers its 130 funcs there via tsp_spu_recomp_register()
 * under spu_begin_image(100)).  Returns the SPU status at halt.
 * ------------------------------------------------------------------------- */
extern const unsigned char g_taskset_policy_bytes[];
extern const unsigned g_taskset_policy_size;
extern void tsp_spu_func_00000A00(spu_context*);

/* pm-flow hang watchdog: dump the traced run's partial transfer trail + live
 * ctx state 5s after arm if the run hasn't returned (g_pm_flow_ctx still set).
 * Reads of the running ctx are racy by design -- diagnostic only. */
#ifdef _WIN32
#include <windows.h>
static DWORD WINAPI spurs_pm_flow_watchdog(LPVOID p)
{
    (void)p;
    extern uint32_t g_pm_flow_buf[]; extern volatile unsigned g_pm_flow_n;
    extern void* volatile g_pm_flow_ctx;
    Sleep(5000);
    spu_context* ctx = (spu_context*)g_pm_flow_ctx;
    if (!ctx) { fprintf(stderr, "[pm-flow] watchdog: run returned (no hang)\n"); return 0; }
    unsigned n = g_pm_flow_n; if (n > 8192) n = 8192;
    fprintf(stderr, "[pm-flow] HUNG-RUN PARTIAL: %u transfers, live pc=0x%05X "
            "LS12A0=%02X%02X LS12C0=%02X%02X LS12F0=%02X%02X%02X%02X LS1300=%02X%02X:",
            n, ctx->pc & SPU_LS_MASK,
            ctx->ls[0x12A0], ctx->ls[0x12A1],
            ctx->ls[0x12C0], ctx->ls[0x12C1],
            ctx->ls[0x12F0], ctx->ls[0x12F1], ctx->ls[0x12F2], ctx->ls[0x12F3],
            ctx->ls[0x1300], ctx->ls[0x1301]);
    /* Pipeline-state hex dump: batch commands (0xC00), per-slot records (0xDF0),
     * staging ring (0xEB0), FFFF markers (0x11B0), scheduler vars (0x1270-0x1340),
     * queue caches (0x13B0-0x1520). One shot -- everything the state-2 cycle's
     * stage decision reads, so the divergent condition can be hand-simulated. */
    { static const struct { uint32_t a, len; const char* tag; } R[] = {
          { 0xC00, 0x60, "cmds" }, { 0xDF0, 0x40, "rec" }, { 0xEB0, 0x40, "ring" },
          { 0x11B0, 0x20, "mark" }, { 0x1270, 0x40, "sched" }, { 0x12C0, 0x50, "state" },
          { 0x13B0, 0x30, "qvar" }, { 0x14A0, 0x30, "qcache" }, { 0x1440, 0x40, "batchrec" } };
      for (unsigned r = 0; r < sizeof R / sizeof R[0]; r++) {
          fprintf(stderr, "\n[pm-ls] %-8s 0x%04X:", R[r].tag, R[r].a);
          for (uint32_t o = 0; o < R[r].len; o += 4) {
              if (o && (o & 15) == 0) fprintf(stderr, " |");
              fprintf(stderr, " %02X%02X%02X%02X",
                      ctx->ls[R[r].a+o], ctx->ls[R[r].a+o+1],
                      ctx->ls[R[r].a+o+2], ctx->ls[R[r].a+o+3]);
          }
      }
      fprintf(stderr, "\n"); }
    uint32_t last = 0xFFFFFFFFu; unsigned rep = 0; unsigned col = 0;
    for (unsigned i = 0; i < n; i++) {
        uint32_t pc = g_pm_flow_buf[i] & SPU_LS_MASK;
        if (pc == last) { rep++; continue; }
        if (rep) { fprintf(stderr, "*%u", rep + 1); rep = 0; }
        fprintf(stderr, "%s%05X", (col++ % 16) ? " " : "\n  ", pc);
        last = pc;
    }
    if (rep) fprintf(stderr, "*%u", rep + 1);
    fprintf(stderr, "\n"); fflush(stderr);
    return 0;
}
void spurs_pm_flow_watchdog_start(void)
{
    HANDLE h = CreateThread(NULL, 0, spurs_pm_flow_watchdog, NULL, 0, NULL);
    if (h) CloseHandle(h);
}
#else
void spurs_pm_flow_watchdog_start(void) {}
#endif

int spurs_run_taskset_policy_probe(uint32_t taskset_ea, uint32_t taskid,
                                   uint32_t spurs_ea, uint64_t wkl_data,
                                   uint32_t wid, uint8_t* out_2700, uint32_t out_len)
{
    spu_context* ctx = (spu_context*)calloc(1, sizeof(spu_context));
    if (!ctx) return -1;
    spu_context_init(ctx, 0);
    ctx->image_id    = 100;      /* policy's own indirect-branch table */
    ctx->policy_mode = 1;

    /* Policy module image at its load base LS 0xA00 (code + data tables). */
    /* size 0 = this port linked the default stub blob (spurs_policy_blob_weak.c),
     * i.e. it has no lifted taskset policy image -- don't run an empty LS. */
    if (!g_taskset_policy_size ||
        g_taskset_policy_size > SPU_LS_SIZE - 0xA00) { free(ctx); return -1; }
    memcpy(ctx->ls + 0xA00, g_taskset_policy_bytes, g_taskset_policy_size);

    /* SpursKernelContext at LS 0x100 (same layout as spu_run_policy_module). */
    uint8_t* ls = ctx->ls;
#define KBE32(off, v) do { uint32_t _v = (v); ls[(off)] = (uint8_t)(_v >> 24); \
        ls[(off)+1] = (uint8_t)(_v >> 16); ls[(off)+2] = (uint8_t)(_v >> 8);   \
        ls[(off)+3] = (uint8_t)_v; } while (0)
    KBE32(0x1C0, 0);                    KBE32(0x1C4, spurs_ea);
    KBE32(0x1C8, 0);                    KBE32(0x1CC, 8);
    KBE32(0x1D0, 0);                    KBE32(0x1D4, 0xA00);
    KBE32(0x1D8, wid);                  KBE32(0x1DC, wid);
    KBE32(0x1E0, SPURS_PM_EXIT_TO_KERNEL_LS);
    KBE32(0x1E4, SPURS_PM_SELECT_WORKLOAD_LS);
    /* taskset EA also planted at the SpursKernelContext taskset slot the policy
     * reads for its taskset DMA (mirrors the cri-r4 injection at LS 0x27B8). */
#undef KBE32

    ctx->gpr[0]._u32[0] = SPURS_PM_EXIT_TO_KERNEL_LS;
    ctx->gpr[1]._u32[0] = 0x3FFB0;
    ctx->gpr[3]._u32[0] = 0x100;
    ctx->gpr[4]._u32[0] = (uint32_t)(wkl_data >> 32);
    ctx->gpr[4]._u32[1] = (uint32_t)wkl_data;
    ctx->gpr[5]._u32[0] = 0;

    g_spurs_pm_polls = 0; g_spurs_pm_exited = 0;
    { extern volatile int g_abx; extern SPU_THREAD_LOCAL int g_spu_bo_reset; if (g_abx & 0x80) g_spu_bo_reset = 1; }   /* v132 */

    fprintf(stderr, "[real-pm] RUN taskset=0x%08X task=%u spurs=0x%08X wkl=0x%llX wid=%u\n",
            taskset_ea, taskid, spurs_ea, (unsigned long long)wkl_data, wid);
    fflush(stderr);

    spu_run_with_halt(tsp_spu_func_00000A00, ctx);

    fprintf(stderr, "[real-pm] END status=0x%X pc=0x%05X polls=%u exited=%u\n",
            ctx->status, ctx->pc, g_spurs_pm_polls, g_spurs_pm_exited);
    fflush(stderr);

    if (out_2700 && out_len) {
        uint32_t n = out_len;
        if ((uint32_t)0x2700 + n > SPU_LS_SIZE) n = SPU_LS_SIZE - 0x2700;
        memcpy(out_2700, ctx->ls + 0x2700, n);
    }
    int st = (int)ctx->status;
    { extern void spu_coh_unregister(spu_context*); spu_coh_unregister(ctx); }
    free(ctx);
    return st;
}


/* ---- WWS_RUNWATCH: stuck policy-module run sampler ---- */
#ifdef _WIN32
#include <windows.h>
#else
#include "../platform/win32_compat.h"
#endif
#define RUNWATCH_N 16
static struct { void* volatile ctx; unsigned wid, sn; unsigned long long t0; int samples; } s_rw[RUNWATCH_N];
static unsigned long long s_rw_ms = 0;
static DWORD WINAPI spurs_runwatch_thread(LPVOID p)
{
    (void)p;
    extern unsigned long long ps3_ms_now(void);
    for (;;) {
        Sleep(1000);
        unsigned long long now = ps3_ms_now();
        for (int i = 0; i < RUNWATCH_N; i++) {
            spu_context* c = (spu_context*)s_rw[i].ctx;
            if (!c || now - s_rw[i].t0 < s_rw_ms || s_rw[i].samples >= 30) continue;
            s_rw[i].samples++;
            const uint8_t* L = c->ls;
            #define W(o) (((uint32_t)L[(o)]<<24)|((uint32_t)L[(o)+1]<<16)|((uint32_t)L[(o)+2]<<8)|L[(o)+3])
            fprintf(stderr, "[runwatch] wid=%u spu=%u stuck %llums: pc=0x%05X r0=0x%05X r3=%08X depth=%u status=0x%X "
                    "ev=0x%X/0x%X ie=%d srr0=0x%05X stallmask=0x%X stallstat=0x%X | loadState=%08X nextLoad=%08X "
                    "load=%08X run=%08X store=%08X LS0=%08X\n",
                    s_rw[i].wid, s_rw[i].sn, now - s_rw[i].t0, (uint32_t)c->pc & SPU_LS_MASK,
                    c->gpr[0]._u32[0] & SPU_LS_MASK, c->gpr[3]._u32[0], c->host_depth, c->status,
                    c->event_status, c->event_mask, c->int_enable, c->srr0 & SPU_LS_MASK,
                    c->list_stall_mask, c->list_stall_stat,
                    W(0x12A0), W(0x12C0), W(0x12D0), W(0x12E0), W(0x12F0), W(0));
            #undef W
            fflush(stderr);
        }
    }
    return 0;
}
int spurs_runwatch_begin(void* ctx, unsigned wid, unsigned sn)
{
    static int s_on = -1;
    if (s_on < 0) {
        const char* e = getenv("WWS_RUNWATCH");
        s_rw_ms = e ? strtoull(e, 0, 0) : 0;
        s_on = s_rw_ms ? 1 : 0;
        if (s_on) { HANDLE h = CreateThread(NULL, 0, spurs_runwatch_thread, NULL, 0, NULL); if (h) CloseHandle(h); }
    }
    if (!s_on) return 0;
    extern unsigned long long ps3_ms_now(void);
    for (int i = 0; i < RUNWATCH_N; i++)
        if (!s_rw[i].ctx) { s_rw[i].wid = wid; s_rw[i].sn = sn; s_rw[i].t0 = ps3_ms_now(); s_rw[i].samples = 0;
                            s_rw[i].ctx = ctx; return 1; }
    return 0;
}
void spurs_runwatch_end(void* ctx)
{
    for (int i = 0; i < RUNWATCH_N; i++) if (s_rw[i].ctx == ctx) { s_rw[i].ctx = 0; return; }
}
