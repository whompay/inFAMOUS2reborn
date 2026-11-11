/*
 * ps3recomp - Timer and time syscalls (implementation)
 */

#include "sys_timer.h"
#include "sys_event.h"
#include "../memory/vm.h"
#include <stdlib.h>   /* getenv -- an implicit decl returns int, truncating the pointer */
#include <string.h>
#include <stdlib.h>

/* ---------------------------------------------------------------------------
 * Globals
 * -----------------------------------------------------------------------*/
sys_timer_info g_sys_timers[SYS_TIMER_MAX];

#ifdef _WIN32
static LARGE_INTEGER s_qpc_freq;
static int           s_qpc_init = 0;

static void ensure_qpc_init(void)
{
    if (!s_qpc_init) {
        QueryPerformanceFrequency(&s_qpc_freq);
        s_qpc_init = 1;
    }
}

int64_t lv2_usec_deadline(uint64_t usec)
{
    LARGE_INTEGER now;
    ensure_qpc_init();
    QueryPerformanceCounter(&now);
    return now.QuadPart +
        (int64_t)((usec * (uint64_t)s_qpc_freq.QuadPart) / 1000000ULL);
}

int lv2_deadline_passed(int64_t deadline)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return now.QuadPart >= deadline;
}
#endif

/* ---------------------------------------------------------------------------
 * PPU timebase (mftb/mftbu) -- THE guest clock.
 *
 * One global monotonic counter scaled to the PS3 timebase (79.8 MHz),
 * anchored at first use. The old lifter emission was a per-call-site
 * static that advanced 16667 ticks PER READ -- not a clock at all: every
 * guest timing loop (media pacers, throttles, profilers) computed garbage
 * from it, and each call site had a PRIVATE counter that only moved when
 * polled. Overflow-safe split multiply (rem < qpf, so
 * rem * 79.8e6 < ~8e14 << 2^63).
 * -----------------------------------------------------------------------*/
uint64_t ppu_timebase_now(void)
{
#ifdef _WIN32
    static LONGLONG t0 = 0;
    ensure_qpc_init();
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    if (!t0) t0 = now.QuadPart;   /* benign race: same anchor either way */
    uint64_t d = (uint64_t)(now.QuadPart - t0);
    uint64_t q = (uint64_t)s_qpc_freq.QuadPart;
    return (d / q) * PS3_TIMEBASE_FREQ + (d % q) * PS3_TIMEBASE_FREQ / q;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * PS3_TIMEBASE_FREQ +
           (uint64_t)ts.tv_nsec * PS3_TIMEBASE_FREQ / 1000000000ull;
#endif
}

static void write_be32(uint32_t addr, uint32_t val)
{
    uint32_t* p = (uint32_t*)vm_to_host(addr);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ || defined(_WIN32)
    val = ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) |
          ((val <<  8) & 0xFF0000) | ((val << 24) & 0xFF000000u);
#endif
    *p = val;
}

static void write_be64(uint32_t addr, uint64_t val)
{
    uint64_t* p = (uint64_t*)vm_to_host(addr);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ || defined(_WIN32)
    val = ((val >> 56) & 0xFFULL) |
          ((val >> 40) & 0xFF00ULL) |
          ((val >> 24) & 0xFF0000ULL) |
          ((val >>  8) & 0xFF000000ULL) |
          ((val <<  8) & 0xFF00000000ULL) |
          ((val << 24) & 0xFF0000000000ULL) |
          ((val << 40) & 0xFF000000000000ULL) |
          ((val << 56) & 0xFF00000000000000ULL);
#endif
    *p = val;
}

/* ---- KP (2026-10-19): usleep poll-site accounting + LR-keyed sleep cap ----------------------
 * Finding (QP-pprof PS3_POLLTOP, steady tutorial): the main thread's biggest wait is NOT the CTaskMgr loops but
 * func_003374C8 (lr 0x001DB7E8: while (obj->state(+0x318) != 4) usleep(10); ~4.5-4.9 s per 10 s). The state is
 * advanced by guest thread 5 (func_00337414 -> func_0033704C state machine), which polls with usleep(table[state])
 * = 900/100/100/500 us (table at 0x6F23C0) -- and guest thread 4 (func_00338CD4) with usleep(500) -- so every
 * hand-off main -> t5 -> w2 -> t5 -> main is quantised by those sleeps (t5 makes ~230 polls per frame).
 *  KP_USLEEP_CAP=0 (default) | 1 | ab : cap the sleep of usleep calls from KP_USLEEP_CAP_LR (hex list; default
 *      337470,33743C,338D1C,338CF4 = the t5/t4 state-machine polls; "all" = every usleep < 1000 us) to
 *      KP_USLEEP_CAP_US (default 20) microseconds. "ab" = only while PS3_ABX_SEQ bit 0x80000 is set (in-run A/B;
 *      [ABFPS] per phase). A shorter poll sleep is what a faster guest CPU would see; the predicate is re-checked
 *      by the guest's own loop, nothing else changes.
 *  KP_USTAT=1 : every 10 s print [KPUS] the poll sites by real wall time (thread, lr, runs, calls, ms), a "run" =
 *      consecutive usleeps of one thread from one poll loop (lr within 0x100 of the first). With QC_TL recording,
 *      every QC_TL_U_STRIDE-th run is a 'U' event (default 64, 1 = every run; wid = guest tid & 0xFF,
 *      a = lr, b = calls, c = last requested us). KPUS accounting remains unsampled. */
#ifdef _WIN32
static int kp_cap_mode(void)
{
    static int m = -1;
    if (m < 0) { const char* e = getenv("KP_USLEEP_CAP"); m = !e ? 0 : (*e == 'a' || *e == 'A') ? 2 : (*e == '1') ? 1 : 0; }
    return m;
}
static unsigned s_kp_cap_us = 20; static uint32_t s_kp_cap_lr[16]; static int s_kp_cap_n = 0, s_kp_cap_all = 0;
static volatile long s_kp_cap_inited = 0;
static void kp_cap_init(void)
{
    static SRWLOCK lk = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&lk);
    if (s_kp_cap_inited) { ReleaseSRWLockExclusive(&lk); return; }
    { const char* e = getenv("KP_USLEEP_CAP_US"); if (e && *e) { long v = atol(e); if (v >= 0 && v <= 1000) s_kp_cap_us = (unsigned)v; } }
    const char* l = getenv("KP_USLEEP_CAP_LR");
    if (!l || !*l) l = "337470,33743C,338D1C,338CF4";
    if (!strcmp(l, "all")) s_kp_cap_all = 1;
    else while (*l && s_kp_cap_n < 16) { char* q; unsigned long v = strtoul(l, &q, 16); if (q == l) break;
        s_kp_cap_lr[s_kp_cap_n++] = (uint32_t)v; l = q; while (*l == ',' || *l == ' ') l++; }
    fprintf(stderr, "[KPUS] KP usleep cap mode %d (0 off, 1 on, 2 = ABX 0x80000), cap %u us, %s%d LRs\n",
            kp_cap_mode(), s_kp_cap_us, s_kp_cap_all ? "ALL usleeps < 1000 us, " : "", s_kp_cap_n);
    InterlockedExchange(&s_kp_cap_inited, 1);
    ReleaseSRWLockExclusive(&lk);
}
static uint64_t kp_cap_apply(const ppu_context* ctx, uint64_t usec)
{
    const int m = kp_cap_mode();
    if (!m) return usec;
    if (m == 2) { extern volatile int g_abx; if (!(g_abx & 0x80000)) return usec; }
    if (!s_kp_cap_inited) kp_cap_init();
    if (usec <= s_kp_cap_us || usec >= 1000) return usec;
    if (s_kp_cap_all) return s_kp_cap_us;
    const uint32_t lr = (uint32_t)ctx->lr;
    for (int i = 0; i < s_kp_cap_n; i++) if (s_kp_cap_lr[i] == lr) return s_kp_cap_us;
    return usec;
}
/* poll-site runs */
static __declspec(thread) uint32_t t_kp_lr = 0xFFFFFFFFu;
static __declspec(thread) LONGLONG t_kp_t0, t_kp_end;
static __declspec(thread) unsigned t_kp_n, t_kp_req;
static __declspec(thread) unsigned t_kp_qc_u_seq;
static unsigned s_kp_qc_u_stride;
typedef struct { uint64_t key; long long ticks; unsigned runs, calls; } KpSite;
#define KP_SITES 64
static KpSite s_kp_site[KP_SITES];
static SRWLOCK s_kp_lk = SRWLOCK_INIT;
static LONGLONG s_kp_win0;
static int kp_ustat_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("KP_USTAT"); on = (e && *e == '1') ? 1 : 0; }
    return on;
}
#ifdef _MSC_VER
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static void kp_run_close(uint32_t tid)
{
    if (t_kp_lr == 0xFFFFFFFFu || !t_kp_n) return;
    { extern int ps3_qc_rec(void); extern void ps3_qc_ext(int, uint32_t, uint32_t, long long, long long, uint32_t, uint32_t, uint32_t);
      if (ps3_qc_rec()) {
          if (!s_kp_qc_u_stride) {
              const char* e = getenv("QC_TL_U_STRIDE");
              long n = e ? atol(e) : 64;
              if (n < 1) n = 1;
              if (n > 65536) n = 65536;
              s_kp_qc_u_stride = (unsigned)n;
          }
          if (++t_kp_qc_u_seq % s_kp_qc_u_stride == 0)
              ps3_qc_ext('U', tid & 0xFFu, 0, t_kp_t0, t_kp_end, t_kp_lr, t_kp_n, t_kp_req);
      } }
    if (!kp_ustat_on()) return;
    const uint64_t key = ((uint64_t)(tid & 0xFFFFu) << 32) | t_kp_lr;
    AcquireSRWLockExclusive(&s_kp_lk);
    int i = 0;
    for (; i < KP_SITES; i++) if (s_kp_site[i].key == key || !s_kp_site[i].runs) break;
    if (i < KP_SITES) { s_kp_site[i].key = key; s_kp_site[i].ticks += t_kp_end - t_kp_t0; s_kp_site[i].runs++; s_kp_site[i].calls += t_kp_n; }
    if (!s_kp_win0) s_kp_win0 = t_kp_end;
    else if (t_kp_end - s_kp_win0 >= s_qpc_freq.QuadPart * 10) {
        const double win = (double)(t_kp_end - s_kp_win0) / (double)s_qpc_freq.QuadPart;
        char b[1400]; int p = snprintf(b, sizeof b, "[KPUS] %.1fs poll sites by wall (tid:lr runs/calls ms):", win);
        for (int k = 0; k < 12; k++) {
            int best = -1;
            for (int j = 0; j < KP_SITES; j++) if (s_kp_site[j].runs && s_kp_site[j].ticks >= 0 && (best < 0 || s_kp_site[j].ticks > s_kp_site[best].ticks)) best = j;
            if (best < 0 || p > (int)sizeof b - 80) break;
            p += snprintf(b + p, sizeof b - p, " | %u:%06X %u/%u %.0f", (unsigned)(s_kp_site[best].key >> 32), (unsigned)(uint32_t)s_kp_site[best].key,
                          s_kp_site[best].runs, s_kp_site[best].calls, 1000.0 * (double)s_kp_site[best].ticks / (double)s_qpc_freq.QuadPart);
            s_kp_site[best].ticks = -s_kp_site[best].ticks - 1;   /* mark printed */
        }
        { extern volatile int g_abx; fprintf(stderr, "%s (abx 0x%X)\n", b, (unsigned)g_abx); }
        memset(s_kp_site, 0, sizeof s_kp_site);
        s_kp_win0 = t_kp_end;
    }
    ReleaseSRWLockExclusive(&s_kp_lk);
}
/* before the sleep: continue or start a run */
static void kp_run_pre(const ppu_context* ctx, LONGLONG now, uint64_t req)
{
    const uint32_t lr = (uint32_t)ctx->lr;
    /* same poll loop (lr within 0x100 of the run's first lr: inFamous 2's CTaskMgr loops alternate 2-3 call sites) and
     * the guest re-checked its predicate within 200 us -> same wait */
    if (t_kp_lr != 0xFFFFFFFFu && (uint32_t)(lr - t_kp_lr + 0x100u) < 0x200u && now - t_kp_end < s_qpc_freq.QuadPart / 5000) {
        t_kp_n++; t_kp_req = (unsigned)req; return; }
    kp_run_close((uint32_t)ctx->thread_id);
    t_kp_lr = lr; t_kp_t0 = now; t_kp_n = 1; t_kp_req = (unsigned)req; t_kp_end = now;
}
#endif

/* ---------------------------------------------------------------------------
 * sys_timer_usleep
 *
 * r3 = microseconds
 * -----------------------------------------------------------------------*/
int64_t sys_timer_usleep(ppu_context* ctx)
{
    uint64_t usec = LV2_ARG_U64(ctx, 0);
    { extern unsigned long long ps3_qpc_us(void);
      static int n=0; if (n < 60 && n++ < 60)   /* F181: read-only once saturated (was a cross-thread RMW per call) */
        fprintf(stderr, "[WAIT] t=%lluus timer_usleep(%llu us) lr=0x%08llX cia=0x%08llX\n", ps3_qpc_us(),
        (unsigned long long)usec, (unsigned long long)ctx->lr, (unsigned long long)ctx->cia); }
    /* PS3_POLLTOP=<seconds>: name the usleep poll sites.
     *
     * A title that is "hung" is usually not blocked on anything -- every thread
     * is spinning on sys_timer_usleep waiting for a flag, and [BLOCKSUM] only
     * says "syscall 141, 12s". This histograms the callers over a window and
     * prints the top ones, so the answer is a guest address rather than a
     * count. Keyed on ctx->lr: a host-stack backtrace through lifted code has
     * no unwind tables and invents chains (see the Guitar Hero III writeup),
     * and lr is the only guest chain that survives.
     *
     * Deliberately unlocked: the counts are a diagnostic, and a lost increment
     * under a race cannot change which site is at the top of a 30k-per-second
     * poll. */
    { static long s_pt = -1;
      if (s_pt < 0) { const char* e = getenv("PS3_POLLTOP");
                      s_pt = e ? (long)strtoul(e, 0, 10) : 0;
                      if (s_pt < 0) s_pt = 0; }
      if (s_pt > 0) {
        enum { PT_MAX = 128 };
        static uint32_t pt_lr[PT_MAX]; static uint32_t pt_n[PT_MAX];
        static uint64_t pt_us[PT_MAX]; static int pt_used = 0;
        static uint32_t pt_tid[PT_MAX];
        static unsigned long long pt_win = 0;
        uint32_t lr = (uint32_t)ctx->lr;
        int i = 0;
        for (; i < pt_used; i++) if (pt_lr[i] == lr) break;
        if (i == pt_used && pt_used < PT_MAX) {
            pt_lr[pt_used] = lr;
            /* The thread matters as much as the site: "who is waiting" is the
             * half that says whether a poll is the engine idling or the main
             * thread stuck. */
            pt_tid[pt_used] = (uint32_t)ctx->thread_id;
            pt_used++;
        }
        if (i < PT_MAX) { pt_n[i]++; pt_us[i] += usec; }
        { extern unsigned long long ps3_qpc_us(void);
          unsigned long long now = ps3_qpc_us();
          if (!pt_win) pt_win = now;
          else if (now - pt_win >= (unsigned long long)s_pt * 1000000ull) {
            fprintf(stderr, "--- usleep poll sites (last %lds) ---\n", s_pt);
            for (int k = 0; k < 8; k++) {
              int best = -1;
              for (int j = 0; j < pt_used; j++)
                if (pt_n[j] && (best < 0 || pt_n[j] > pt_n[best])) best = j;
              if (best < 0) break;
              fprintf(stderr, "  %8u calls  %6llums slept  tid=%-3u lr=0x%08X\n",
                      pt_n[best], (unsigned long long)(pt_us[best] / 1000),
                      pt_tid[best], pt_lr[best]);
              pt_n[best] = 0;
            }
            fflush(stderr);
            for (int j = 0; j < pt_used; j++) { pt_n[j] = 0; pt_us[j] = 0; }
            pt_win = now;
          } } } }

    /* PS3_WAIT_OBJ=<lr-hex>: when a usleep spin is reached from this return
     * address, dump the registers and the object they point at. A poll loop
     * tells you WHERE it is spinning; this tells you WHAT it is spinning on,
     * which is the part you actually need to find who never releases it. */
    { static long s_wo = -1, s_wo_after = -1;
      static unsigned long long s_wo_t0 = 0;
      extern unsigned long long ps3_qpc_us(void);
      if (s_wo < 0) { const char* e = getenv("PS3_WAIT_OBJ");
                      s_wo = e ? (long)strtoul(e, 0, 16) : 0;
                      e = getenv("PS3_WAIT_OBJ_AFTER");
                      s_wo_after = e ? (long)strtoul(e, 0, 10) : 0;
                      s_wo_t0 = ps3_qpc_us(); }
      if (s_wo && (uint32_t)ctx->lr == (uint32_t)s_wo &&
          ps3_qpc_us() - s_wo_t0 >= (unsigned long long)s_wo_after * 1000000ull) {
        static int _n = 0;
        if (_n++ < 8) {
          /* Dump the words at EVERY plausible object register, not just r29.
           * Which register holds the object is per-title -- r29 was right for
           * the title this was written for and is a spin COUNTER in Guitar
           * Hero III, where the object is in r30/r31. Printing one guess makes
           * the tool silently report the wrong memory as "the thing it waits
           * on", which is worse than printing nothing. */
          fprintf(stderr, "[wait-obj] lr=0x%08X sleep=%lluus", (uint32_t)ctx->lr,
                  (unsigned long long)usec);
          static const int regs[] = { 28, 29, 30, 31, 3 };
          for (unsigned ri = 0; ri < sizeof regs / sizeof regs[0]; ri++) {
            const int r = regs[ri];
            const uint32_t o = (uint32_t)ctx->gpr[r];
            fprintf(stderr, "  r%d=0x%08X", r, o);
            /* Only deref something that looks like a guest pointer: a small
             * integer is a count, and dumping "memory at 4" is noise. */
            if (vm_base && o >= 0x10000u) {
              fprintf(stderr, "[");
              for (int w = 0; w < 4; w++) {
                const uint8_t* q = vm_base + o + (unsigned)w * 4u;
                uint32_t v = ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16) |
                             ((uint32_t)q[2] << 8)  | (uint32_t)q[3];
                fprintf(stderr, "%s%08X", w ? " " : "", v);
              }
              fprintf(stderr, "]");
            }
          }
          /* Queue/state-machine detail for waiters that keep their object in
           * r31.  The state-0 transition uses the active slot, two counter
           * pointers, and the per-slot sequence words.  This remains behind
           * PS3_WAIT_OBJ, so it has no normal-run cost. */
          {
            const uint32_t o = (uint32_t)ctx->gpr[31];
            if (vm_base && o >= 0x10000u && o < 0x70000000u) {
#define WAIT_BE32(a) (((uint32_t)vm_base[(a)] << 24) | \
                      ((uint32_t)vm_base[(a) + 1] << 16) | \
                      ((uint32_t)vm_base[(a) + 2] << 8) | \
                      (uint32_t)vm_base[(a) + 3])
              const uint32_t slot = WAIT_BE32(o + 0x110);
              const uint32_t p184 = WAIT_BE32(o + 0x184);
              const uint32_t p188 = WAIT_BE32(o + 0x188);
              const uint32_t v184 = (p184 >= 0x10000u && p184 < 0x70000000u)
                                      ? WAIT_BE32(p184) : 0;
              const uint32_t v188 = (p188 >= 0x10000u && p188 < 0x70000000u)
                                      ? WAIT_BE32(p188) : 0;
              fprintf(stderr,
                      "  obj31{slot=%u p184=%08X:%08X p188=%08X:%08X "
                      "seq0=%08X seq1=%08X q300=%08X state=%08X}",
                      slot, p184, v184, p188, v188,
                      WAIT_BE32(o + 0x190), WAIT_BE32(o + 0x194),
                      WAIT_BE32(o + 0x300), WAIT_BE32(o + 0x318));
#undef WAIT_BE32
            }
          }
          fprintf(stderr, "\n"); fflush(stderr);
        } } }

    /* POLLSITE: resolve the host chain of the 1ms poller (LBP bringup) to
     * guest functions -- names the stage that is starving. */
    { static int _ps = -1; if (_ps < 0) _ps = getenv("POLLSITE") ? 12 : 0;
      if (_ps > 0 && usec == 1000) { _ps--;
        extern void ppu_log_host_chain(const char*);
        ppu_log_host_chain("usleep1ms"); } }

#ifdef _WIN32
    /* QP (2026-10-16): main-thread scheduling A/B switch (ppu_loader.cpp ps3_qp_main_tick; QP_MAIN_PCORE=ab). */
    { extern volatile unsigned long g_qp_main_tid; extern void ps3_qp_main_tick(void);
      if (g_qp_main_tid && GetCurrentThreadId() == g_qp_main_tid) ps3_qp_main_tick(); }
#endif
#ifdef _WIN32
    /* KP (2026-10-19) KP_PCORE_TIDS=<guest tid,...> (default unset = off): those guest threads (inFamous 2: 4,5 = the
     * job-dispatch state machines the main thread waits on, unpinned NORMAL -> E-cores) run on the P-cores at
     * ABOVE_NORMAL. KP_PCORE=1 always, =ab (default) only while PS3_ABX_SEQ bit 0x100000 is set. Applied/undone
     * from the thread's own usleep (the only place these poll threads reliably pass). */
    { static int s_kpc = -2; static uint32_t s_kpc_mask = 0;
      if (s_kpc == -2) { const char* e = getenv("KP_PCORE_TIDS"); int m = -1;
          if (e && *e) { const char* q = e; while (*q) { int t = atoi(q); if (t >= 0 && t < 32) s_kpc_mask |= 1u << t;
                  while (*q && *q != ',') q++; while (*q == ',' || *q == ' ') q++; }
              const char* k = getenv("KP_PCORE"); m = (k && *k == '1') ? 1 : 2; }
          s_kpc = s_kpc_mask ? m : -1; }
      if (s_kpc > 0) {
          static __declspec(thread) int t_kpc = 0;
          const uint32_t gt = (uint32_t)ctx->thread_id;
          if (gt < 32 && ((s_kpc_mask >> gt) & 1)) {
              extern volatile int g_abx;
              const int want = (s_kpc == 1) || (g_abx & 0x100000) != 0;
              if (want != t_kpc) { extern void ps3_thread_pcore_set(const char*, int, int);
                  t_kpc = want; ps3_thread_pcore_set("KP guest poll thread", want, THREAD_PRIORITY_ABOVE_NORMAL); }
          }
      } }
#endif
    /* v82 SPURS_KICK: a short guest sleep is a completion poll (CTaskMgr job wait at lr 002D0054):
     * wake SPURS kernel threads parked in their idle wait so the job is picked up now. */
    { extern int ps3_spurs_kick_us(void); extern void ps3_spurs_kick(void);
      static int s_kus = -2; if (s_kus == -2) s_kus = ps3_spurs_kick_us();
      if (s_kus >= 0 && usec <= (uint64_t)s_kus) ps3_spurs_kick(); }
    /* NB16 ABX 0x1000 (PS3_ABX_SEQ): in-run A/B of the v82 kick -- a guest usleep <= NB_KICK_US (default 5) wakes
     * parked SPURS kernel threads, at most once per NB_KICK_GAP_US (default 50) per thread (the CTaskMgr spin polls
     * ~200k/s; ps3_spurs_kick is one read when nobody is parked). */
    { extern volatile int g_abx;
      if (g_abx & 0x1000) {
        static int s_nk = -2, s_ng = 50;
        if (s_nk == -2) { const char* e = getenv("NB_KICK_US"); s_nk = e ? atoi(e) : 5;
                          e = getenv("NB_KICK_GAP_US"); s_ng = e ? atoi(e) : 50; if (s_ng < 0) s_ng = 0; }
        if (usec <= (uint64_t)(s_nk < 0 ? 0 : s_nk)) {
          extern unsigned long long ps3_qpc_us(void);
#ifdef _WIN32
          static __declspec(thread) unsigned long long t_nk_last = 0;
#else
          static __thread unsigned long long t_nk_last = 0;
#endif
          const unsigned long long nw = ps3_qpc_us();
          if (nw - t_nk_last >= (unsigned long long)s_ng) { t_nk_last = nw; extern void ps3_spurs_kick(void); ps3_spurs_kick(); }
        } } }

#ifdef _WIN32
    /* KP (2026-10-19): poll-site runs + LR-keyed sleep cap (see the KP block above sys_timer_usleep) */
    usec = kp_cap_apply(ctx, usec);
    int kp_tr = 0;
    { extern int ps3_qc_rec(void);
      if (kp_ustat_on() || ps3_qc_rec()) { kp_tr = 1; ensure_qpc_init(); LARGE_INTEGER kq; QueryPerformanceCounter(&kq); kp_run_pre(ctx, kq.QuadPart, usec); } }
    /* Use high-resolution sleep via waitable timer for better precision */
    if (usec >= 1000) {
        /* v59: one HIGH-RESOLUTION waitable timer per thread (Win10 1803+), created once.
         * The old code created a plain timer per call: ~1-2 ms granularity, so the game's
         * audio driver (brbDriver polls with usleep(1000) ~800x/s) and every other 1 ms
         * poller slept up to twice as long as asked -- late audio blocks = stutter. */
        static __declspec(thread) HANDLE t_timer = NULL;
        static __declspec(thread) int t_timer_tried = 0;
        if (!t_timer_tried) {
            t_timer_tried = 1;
            t_timer = CreateWaitableTimerExW(NULL, NULL, 0x00000002 /*CREATE_WAITABLE_TIMER_HIGH_RESOLUTION*/,
                                             0x1F0003 /*TIMER_ALL_ACCESS*/);
            if (!t_timer) t_timer = CreateWaitableTimerW(NULL, TRUE, NULL);
        }
        if (t_timer) {
            LARGE_INTEGER due;
            due.QuadPart = -((LONGLONG)usec * 10); /* 100ns units, negative = relative */
            SetWaitableTimer(t_timer, &due, 0, NULL, NULL, FALSE);
            WaitForSingleObject(t_timer, INFINITE);
        } else {
            Sleep((DWORD)(usec / 1000));
        }
    } else if (usec > 0) {
        /* Sub-millisecond sleep. Sleep()/waitable timers round up to the
         * ~0.5-1 ms scheduler quantum, far too coarse for a guest that relies
         * on accurate microsecond pacing -- a short usleep() spin waiting on
         * another thread to make progress otherwise collapses to a single yield
         * and busy-spins at full speed (so the wait is effectively a no-op).
         * v54: the old loop called SwitchToThread even for the game's 1-5 us
         * job-counter polls.  A scheduler yield commonly took 9-20 us on the
         * target machine, and the main PPU thread spent about 30% of wall time
         * here.  For genuinely tiny sleeps use PAUSE instead: the guest checks
         * its predicate again at the requested deadline, while PAUSE remains
         * friendly to an SMT sibling.  Longer sub-ms waits still yield so they
         * cannot monopolise a core.  PS3_SHORT_SLEEP_SPIN=0 restores v53. */
        ensure_qpc_init();
        LARGE_INTEGER qpc_start, qpc_now;
        QueryPerformanceCounter(&qpc_start);
        const int64_t qpc_deadline = qpc_start.QuadPart +
            (int64_t)((usec * (uint64_t)s_qpc_freq.QuadPart) / 1000000ULL);
        static int s_short_spin = -1;
        if (s_short_spin < 0) {
            const char* e = getenv("PS3_SHORT_SLEEP_SPIN");
            s_short_spin = (!e || *e != '0') ? 1 : 0;
            fprintf(stderr, "[timer] <=50us sleeps use %s (PS3_SHORT_SLEEP_SPIN)\n",
                    s_short_spin ? "QPC+PAUSE" : "QPC+SwitchToThread (v53)");
        }
        /* inFamous 2's main thread waits in func_003374C8 for r31+0x318 to
         * become 4, sleeping 10 us between checks.  Observe the same predicate
         * during the existing deadline spin so completion can end the final
         * sleep early; an unchanged predicate still gets the full timeout. */
        static int s_usleep_fast = -1;
        volatile const uint32_t* fast_state = NULL;
        if (s_usleep_fast < 0) {
            const char* e = getenv("USLEEP_FAST");
            s_usleep_fast = (e && *e == '1') ? 1 : ((e && (*e == 'a' || *e == 'A')) ? 2 : 0);   /* HP3PPU: ab = ABX 0x4000000 */
            if (s_usleep_fast)
                fprintf(stderr, "[timer] USLEEP_FAST=%s: main state wait polls r31+0x318 during <=50us sleeps (ON)\n",
                        s_usleep_fast == 2 ? "ab (only while PS3_ABX_SEQ bit 0x4000000)" : "1");
        }
        extern volatile int g_abx;
        if (s_usleep_fast && (s_usleep_fast == 1 || (g_abx & 0x4000000)) && usec <= 50 && (uint32_t)ctx->lr == 0x001DB7E8u) {
            extern volatile unsigned long g_qp_main_tid;
            const uint32_t obj = (uint32_t)ctx->gpr[31];
            if (g_qp_main_tid && GetCurrentThreadId() == g_qp_main_tid &&
                vm_base && obj >= 0x10000u && obj < 0x70000000u - 0x318u)
                fast_state = (volatile const uint32_t*)(vm_base + obj + 0x318u);
        }
        extern volatile int g_ab_mask;   /* v66 PS3_AB bit0 flips PS3_SHORT_SLEEP_SPIN */
        if ((fast_state || (s_short_spin ^ ((g_ab_mask >> 4) & 1))) && usec <= 50) {   /* v67: bit4 (unused in the v67 cycle) */
            do {
                if (fast_state) {
                    const uint32_t raw = *fast_state;
                    const uint32_t state = ((raw & 0x000000FFu) << 24) |
                                           ((raw & 0x0000FF00u) << 8) |
                                           ((raw & 0x00FF0000u) >> 8) |
                                           ((raw & 0xFF000000u) >> 24);
                    if (state == 4u) break;
                }
                YieldProcessor();
                QueryPerformanceCounter(&qpc_now);
            } while (qpc_now.QuadPart < qpc_deadline);
        } else {
            do {
                SwitchToThread();
                QueryPerformanceCounter(&qpc_now);
            } while (qpc_now.QuadPart < qpc_deadline);
        }
    }
    if (kp_tr) { LARGE_INTEGER kq; QueryPerformanceCounter(&kq); t_kp_end = kq.QuadPart; }   /* KP */
#else
    if (usec > 0) {
        struct timespec ts;
        ts.tv_sec  = (time_t)(usec / 1000000);
        ts.tv_nsec = (long)((usec % 1000000) * 1000);
        nanosleep(&ts, NULL);
    }
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_sleep
 *
 * r3 = seconds
 * -----------------------------------------------------------------------*/
int64_t sys_timer_sleep(ppu_context* ctx)
{
    uint32_t sec = LV2_ARG_U32(ctx, 0);

#ifdef _WIN32
    Sleep(sec * 1000);
#else
    sleep(sec);
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_time_get_current_time
 *
 * r3 = pointer to receive seconds (u64*)
 * r4 = pointer to receive nanoseconds (u64*)
 * -----------------------------------------------------------------------*/
int64_t sys_time_get_current_time(ppu_context* ctx)
{
    uint32_t sec_addr  = LV2_ARG_PTR(ctx, 0);
    uint32_t nsec_addr = LV2_ARG_PTR(ctx, 1);

    uint64_t sec, nsec;

#ifdef _WIN32
    ensure_qpc_init();
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);

    /* Convert QPC to seconds + nanoseconds */
    sec  = (uint64_t)(now.QuadPart / s_qpc_freq.QuadPart);
    uint64_t remainder = (uint64_t)(now.QuadPart % s_qpc_freq.QuadPart);
    nsec = (remainder * 1000000000ULL) / (uint64_t)s_qpc_freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    sec  = (uint64_t)ts.tv_sec;
    nsec = (uint64_t)ts.tv_nsec;
#endif

    if (sec_addr != 0) {
        write_be64(sec_addr, sec);
    }
    if (nsec_addr != 0) {
        write_be64(nsec_addr, nsec);
    }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_time_get_timebase_frequency
 *
 * Returns the PS3 timebase frequency in r3.
 * -----------------------------------------------------------------------*/
int64_t sys_time_get_timebase_frequency(ppu_context* ctx)
{
    (void)ctx;
    return (int64_t)PS3_TIMEBASE_FREQ;
}

/* ---------------------------------------------------------------------------
 * Periodic timer thread
 * -----------------------------------------------------------------------*/

/* Forward declaration */
static void timer_send_event(sys_timer_info* t);

#ifdef _WIN32
static DWORD WINAPI timer_thread_proc(LPVOID param)
{
    sys_timer_info* t = (sys_timer_info*)param;

    while (t->running) {
        DWORD ms = (DWORD)(t->period_usec / 1000);
        if (ms == 0) ms = 1;

        DWORD result = WaitForSingleObject(t->stop_event, ms);
        if (result == WAIT_OBJECT_0) {
            break; /* stop signaled */
        }

        if (t->running) {
            timer_send_event(t);
        }
    }

    return 0;
}
#else
static void* timer_thread_proc(void* param)
{
    sys_timer_info* t = (sys_timer_info*)param;

    pthread_mutex_lock(&t->mtx);
    while (!t->stop_flag) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec  += (time_t)(t->period_usec / 1000000);
        ts.tv_nsec += (long)((t->period_usec % 1000000) * 1000);
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000L;
        }

        int rc = pthread_cond_timedwait(&t->cv, &t->mtx, &ts);
        if (t->stop_flag) break;

        if (rc == ETIMEDOUT) {
            pthread_mutex_unlock(&t->mtx);
            timer_send_event(t);
            pthread_mutex_lock(&t->mtx);
        }
    }
    pthread_mutex_unlock(&t->mtx);

    return NULL;
}
#endif

static void timer_send_event(sys_timer_info* t)
{
    if (t->event_queue_id <= 0 || t->event_queue_id > SYS_EVENT_QUEUE_MAX)
        return;

    sys_event_queue_info* q = &g_sys_event_queues[t->event_queue_id - 1];
    if (!q->active)
        return;

    sys_event_t evt;
    evt.source = t->source;
    evt.data1  = t->data1;
    evt.data2  = 0;
    evt.data3  = 0;

    /* Push event (ignore if queue full) */
#ifdef _WIN32
    EnterCriticalSection(&q->lock);
#else
    pthread_mutex_lock(&q->lock);
#endif

    if (q->count < q->capacity) {
        q->buffer[q->tail] = evt;
        q->tail = (q->tail + 1) % q->capacity;
        q->count++;
#ifdef _WIN32
        WakeConditionVariable(&q->not_empty);
#else
        pthread_cond_signal(&q->not_empty);
#endif
    }

#ifdef _WIN32
    LeaveCriticalSection(&q->lock);
#else
    pthread_mutex_unlock(&q->lock);
#endif
}

/* ---------------------------------------------------------------------------
 * sys_timer_create
 *
 * r3 = pointer to receive timer ID (u32*)
 * -----------------------------------------------------------------------*/
int64_t sys_timer_create(ppu_context* ctx)
{
    uint32_t id_out_addr = LV2_ARG_PTR(ctx, 0);

    int slot = -1;
    for (int i = 0; i < SYS_TIMER_MAX; i++) {
        if (!g_sys_timers[i].active) { slot = i; break; }
    }
    if (slot < 0)
        return (int64_t)(int32_t)CELL_EAGAIN;

    sys_timer_info* t = &g_sys_timers[slot];
    memset(t, 0, sizeof(*t));
    t->active  = 1;
    t->running = 0;
    t->event_queue_id = 0;

#ifdef _WIN32
    t->stop_event    = NULL;
    t->thread_handle = NULL;
    t->timer_handle  = NULL;
#else
    pthread_mutex_init(&t->mtx, NULL);
    pthread_cond_init(&t->cv, NULL);
    t->stop_flag = 0;
#endif

    uint32_t timer_id = (uint32_t)(slot + 1);
    if (id_out_addr != 0) {
        write_be32(id_out_addr, timer_id);
    }

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_destroy
 *
 * r3 = timer_id
 * -----------------------------------------------------------------------*/
int64_t sys_timer_destroy(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    /* Stop if running */
    if (t->running) {
        t->running = 0;
#ifdef _WIN32
        if (t->stop_event) SetEvent(t->stop_event);
        if (t->thread_handle) {
            WaitForSingleObject(t->thread_handle, 3000);
            CloseHandle(t->thread_handle);
        }
        if (t->stop_event) CloseHandle(t->stop_event);
#else
        pthread_mutex_lock(&t->mtx);
        t->stop_flag = 1;
        pthread_cond_signal(&t->cv);
        pthread_mutex_unlock(&t->mtx);
        pthread_join(t->thread, NULL);
        pthread_mutex_destroy(&t->mtx);
        pthread_cond_destroy(&t->cv);
#endif
    }

    t->active = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_connect_event_queue
 *
 * r3 = timer_id
 * r4 = queue_id
 * r5 = source
 * r6 = data1
 * -----------------------------------------------------------------------*/
int64_t sys_timer_connect_event_queue(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);
    uint32_t queue_id = LV2_ARG_U32(ctx, 1);
    uint64_t source   = LV2_ARG_U64(ctx, 2);
    uint64_t data1    = LV2_ARG_U64(ctx, 3);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    if (queue_id == 0 || queue_id > SYS_EVENT_QUEUE_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    t->event_queue_id = (int32_t)queue_id;
    t->source         = source;
    t->data1          = data1;

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_disconnect_event_queue
 *
 * r3 = timer_id
 * -----------------------------------------------------------------------*/
int64_t sys_timer_disconnect_event_queue(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    t->event_queue_id = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_start
 *
 * r3 = timer_id
 * r4 = base_time (absolute start time, 0 = now)
 * r5 = period_usec
 * -----------------------------------------------------------------------*/
int64_t sys_timer_start(ppu_context* ctx)
{
    uint32_t timer_id    = LV2_ARG_U32(ctx, 0);
    /* uint64_t base_time = LV2_ARG_U64(ctx, 1); -- ignored for now */
    uint64_t period      = LV2_ARG_U64(ctx, 2);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    if (t->running)
        return (int64_t)(int32_t)CELL_EBUSY;

    if (period == 0)
        return (int64_t)(int32_t)CELL_EINVAL;

    t->period_usec = period;
    t->running     = 1;

#ifdef _WIN32
    t->stop_event    = CreateEventA(NULL, TRUE, FALSE, NULL);
    t->thread_handle = CreateThread(NULL, 0, timer_thread_proc, t, 0, NULL);
#else
    t->stop_flag = 0;
    pthread_create(&t->thread, NULL, timer_thread_proc, t);
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * sys_timer_stop
 *
 * r3 = timer_id
 * -----------------------------------------------------------------------*/
int64_t sys_timer_stop(ppu_context* ctx)
{
    uint32_t timer_id = LV2_ARG_U32(ctx, 0);

    if (timer_id == 0 || timer_id > SYS_TIMER_MAX)
        return (int64_t)(int32_t)CELL_ESRCH;

    sys_timer_info* t = &g_sys_timers[timer_id - 1];
    if (!t->active)
        return (int64_t)(int32_t)CELL_ESRCH;

    if (!t->running)
        return CELL_OK;

    t->running = 0;

#ifdef _WIN32
    if (t->stop_event) SetEvent(t->stop_event);
    if (t->thread_handle) {
        WaitForSingleObject(t->thread_handle, 3000);
        CloseHandle(t->thread_handle);
        t->thread_handle = NULL;
    }
    if (t->stop_event) {
        CloseHandle(t->stop_event);
        t->stop_event = NULL;
    }
#else
    pthread_mutex_lock(&t->mtx);
    t->stop_flag = 1;
    pthread_cond_signal(&t->cv);
    pthread_mutex_unlock(&t->mtx);
    pthread_join(t->thread, NULL);
#endif

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Registration
 *
 * NOTE: There are syscall number collisions in the existing table:
 *   SYS_TIMER_USLEEP (141) == SYS_EVENT_FLAG_WAIT (141)
 *   SYS_TIMER_SLEEP  (142) == SYS_EVENT_FLAG_TRYWAIT (142)
 *   SYS_TIME_GET_CURRENT_TIME (145) == SYS_EVENT_FLAG_CANCEL (145)
 * The event flag handlers are registered separately and will win.
 * Timer sleep/time functions should be called via wrapper functions
 * or fixed syscall numbers should be assigned.
 * For now, we register the timer-specific syscalls that don't collide.
 * -----------------------------------------------------------------------*/
void sys_timer_init(lv2_syscall_table* tbl)
{
    memset(g_sys_timers, 0, sizeof(g_sys_timers));

#ifdef _WIN32
    ensure_qpc_init();
#endif

    lv2_syscall_register(tbl, SYS_TIMER_CREATE,                   sys_timer_create);
    lv2_syscall_register(tbl, SYS_TIMER_DESTROY,                  sys_timer_destroy);
    lv2_syscall_register(tbl, SYS_TIMER_START,                    sys_timer_start);
    lv2_syscall_register(tbl, SYS_TIMER_STOP,                     sys_timer_stop);
    lv2_syscall_register(tbl, SYS_TIMER_CONNECT_EVENT_QUEUE,      sys_timer_connect_event_queue);
    lv2_syscall_register(tbl, SYS_TIMER_DISCONNECT_EVENT_QUEUE,   sys_timer_disconnect_event_queue);

    /* These have conflicting numbers with event flag syscalls.
     * Register them here -- last registration wins. If events are
     * registered after timers, the event handlers will override.
     * The runtime should dispatch timer_usleep/sleep/get_current_time
     * via direct function calls instead. */
    lv2_syscall_register(tbl, SYS_TIME_GET_TIMEBASE_FREQUENCY, sys_time_get_timebase_frequency);

    /* Register these but be aware of collisions */
    lv2_syscall_register(tbl, SYS_TIMER_USLEEP,            sys_timer_usleep);
    lv2_syscall_register(tbl, SYS_TIMER_SLEEP,             sys_timer_sleep);
    lv2_syscall_register(tbl, SYS_TIME_GET_CURRENT_TIME,   sys_time_get_current_time);
}
