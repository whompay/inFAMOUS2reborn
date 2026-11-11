/*
 * ps3recomp - sysPrxForUser CRT (boot-critical HLE)
 *
 * The first firmware functions a PS3 program calls at startup come from
 * sysPrxForUser (the libc/CRT bridge). Some need the full ppu_context (e.g.
 * sys_initialize_tls sets the thread pointer r13), so they register as
 * context-aware handlers (ps3_hle_register_ctx) rather than through the generic
 * integer-ABI table.
 *
 * NIDs are computed from the names (ps3_compute_nid), so this stays correct
 * without hand-written NID literals.
 */
#include "ppu_recomp.h"     /* ppu_context */
#include "ps3emu/nid.h"     /* ps3_compute_nid */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* win32_compat.h is <windows.h> on Windows (CRITICAL_SECTION for the real
 * lwmutex exclusion) and the POSIX shims elsewhere -- Sleep, DWORD, QPC. */
#include "../platform/win32_compat.h"
#include "../memory/vm.h"   /* VM_HLE_INJECT_BASE -- not platform-specific */

extern "C" uint8_t* vm_base;
extern "C" void ps3_hle_register_ctx(uint32_t nid, const char* name, void (*fn)(ppu_context*));
extern "C" uint32_t vm_read32(uint64_t a);
extern "C" uint64_t vm_read64(uint64_t a);
extern "C" void     vm_write32(uint64_t a, uint32_t v);
extern "C" void     vm_write64(uint64_t a, uint64_t v);

/* Simple bump allocator for TLS areas, in a free vm region below the stack. */
static uint32_t s_tls_next = 0x0E000000u;

/* sys_initialize_tls(u64 main_thread_id, u32 tls_seg_addr, u32 tls_seg_size,
 *                     u32 tls_mem_size) -- set up the main thread's TLS block
 * and point r13 (the PPC64 thread pointer) at it. TLS variables are accessed
 * at r13 - 0x7000 (the static TLS block bias). */
static void sys_initialize_tls(ppu_context* ctx)
{
    uint32_t seg_addr = (uint32_t)ctx->gpr[4];
    uint32_t seg_size = (uint32_t)ctx->gpr[5];
    uint32_t mem_size = (uint32_t)ctx->gpr[6];

    uint32_t block = s_tls_next;
    uint32_t total = ((mem_size + 0x7000u + 0x1000u) + 0xFFFu) & ~0xFFFu;
    s_tls_next += total;

    if (seg_addr && seg_size) memcpy(vm_base + block, vm_base + seg_addr, seg_size);
    if (mem_size > seg_size)  memset(vm_base + block + seg_size, 0, mem_size - seg_size);

    ctx->gpr[13] = block + 0x7000u;   /* thread pointer; TLS data at r13-0x7000 */
    ctx->gpr[3]  = 0;                  /* CELL_OK */
    fprintf(stderr, "[crt] sys_initialize_tls: block 0x%08X, r13=0x%08X (seg 0x%X+%u, mem %u)\n",
            block, (uint32_t)ctx->gpr[13], seg_addr, seg_size, mem_size);
}

/* sys_time_get_system_time() -> microseconds since boot, REAL time.
 * This was a fake counter advancing 1 ms PER CALL ("so callers see time
 * progress") -- so any guest clock built on it ran at call-rate, not
 * wall-time. LBP's Bink movie clock (sysGetSystemTime import 0x8461E528
 * lands here) paced the intro at ~1/10th speed: video decoded at <1 fps,
 * the audio preload threshold took forever to fill (movie stayed silent),
 * and every frontend animation timed off it crawled. */
static void sys_time_get_system_time(ppu_context* ctx)
{
#ifdef _WIN32
    static LARGE_INTEGER s_freq, s_base;
    if (!s_freq.QuadPart) { QueryPerformanceFrequency(&s_freq); QueryPerformanceCounter(&s_base); }
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    ctx->gpr[3] = (uint64_t)((now.QuadPart - s_base.QuadPart) * 1000000ull / (uint64_t)s_freq.QuadPart);
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    ctx->gpr[3] = (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
#endif
}

/* sys_process_is_stack(u32 addr) -> 1 if addr is in the stack region. We model
 * a single stack just below the TLS region; good enough for boot checks. */
static void sys_process_is_stack(ppu_context* ctx)
{
    uint32_t a = (uint32_t)ctx->gpr[3];
    ctx->gpr[3] = (a >= 0x0E000000u && a < 0x10000000u) ? 1 : 0;
}

/* ---------------------------------------------------------------------------
 * Lightweight mutex (sys_lwmutex) — sysPrxForUser.
 *
 * The CRT guards global/singleton initialization with lwmutexes. If create is
 * a no-op that never initializes the structure, the guarded init is skipped
 * and the protected registry is left with null function pointers (the early
 * boot then spins calling a null vtable entry). We model the structure for
 * real; locking is a no-op owner stamp (the boot is single-threaded).
 *
 * sys_lwmutex_t (big-endian, 24 bytes):
 *   +0x00 owner (u32)   +0x04 waiter (u32)   +0x08 attribute (u32)
 *   +0x0C recursive_count (u32)   +0x10 sleep_queue (u32)   +0x14 pad
 * sys_lwmutex_attribute_t: +0x00 protocol  +0x04 recursive  +0x08 name[8]
 * -----------------------------------------------------------------------*/
#define LWM_OWNER  0x00
#define LWM_ATTR   0x08
#define LWM_RECUR  0x0C
/* Owner id = ctx->thread_id, which is nonzero for every thread since main
 * registers as id 1 (ppu_thread_register_main). The old 0->1 fallback made
 * main alias the FIRST CREATED thread: each passed the other's recursive
 * re-lock check, both "owned" the lock, and (LBP) main + bringup emitted GCM
 * concurrently -- fences vanished mid-ring. A zero id now means an
 * unregistered context (bug); stamp a sentinel that matches no real thread. */
#define LWM_SELF(ctx) ((uint32_t)(ctx)->thread_id ? (uint32_t)(ctx)->thread_id : 0x7FFFFFFEu)

#if 1
static HANDLE lwm_sem(uint32_t addr);   /* fwd (defined below) */
struct LwmSlot; static struct LwmSlot* lwm_slot(uint32_t addr); static int lwm_fast(void);
static void lwm_word_reset(struct LwmSlot* sl);   /* v96b fwd */
#endif
static void sys_lwmutex_create(ppu_context* ctx)
{
    uint32_t lwm  = (uint32_t)ctx->gpr[3];
    uint32_t attr = (uint32_t)ctx->gpr[4];
    { static long long _n=0; _n++;
      if (getenv("LWM_COUNT") && (_n<=24 || (_n%50000)==0))
        fprintf(stderr, "[LWM] create #%lld lwm=0x%08X attr=0x%08X\n", _n, lwm, attr); }
    uint32_t protocol = attr ? vm_read32(attr + 0) : 0;
    vm_write32(lwm + 0x00, 0);          /* owner */
    vm_write32(lwm + 0x04, 0);          /* waiter */
    vm_write32(lwm + LWM_ATTR, protocol);
    vm_write32(lwm + LWM_RECUR, 0);     /* recursive_count */
    vm_write32(lwm + 0x10, 0);          /* sleep_queue */
    vm_write32(lwm + 0x14, 0);
#if 1 /* lwmutex/lwcond: real on every host (win32_compat.h shims the Win32 sync API off Windows) */
    /* A recreate at a reused address must not inherit a locked slot (e.g. the
     * previous holder exited while holding). Force the semaphore signaled;
     * over-release of an already-free sem fails harmlessly at max count 1. */
    if (lwm_fast()) { struct LwmSlot* sl = lwm_slot(lwm); if (sl) lwm_word_reset(sl); }
    else { HANDLE s = lwm_sem(lwm); if (s) ReleaseSemaphore(s, 1, NULL); }
#endif
    ctx->gpr[3] = 0;
}
/* REAL mutual exclusion. The old no-op ("boot is single-threaded") corrupted
 * every lwmutex-protected structure once LBP spun up its worker/loader threads --
 * notably the dlmalloc mspace behind the game's big-allocator, whose tree then
 * fell apart and reported OOM on a tiny request with 100+ MB free.
 *
 * Backed by a binary SEMAPHORE, not a CRITICAL_SECTION: a CS may only be
 * released by its owning thread, but guest code passes lwmutex ownership
 * between threads (LBP's job system) and threads exit while holding -- one
 * cross-thread unlock silently failed and the still-owned CS parked the next
 * locker forever (the Network-node hang). A semaphore releases from any
 * thread. Recursion is handled explicitly via the guest owner/recur fields
 * we stamp (only the holder ever writes owner=self, so the re-lock check is
 * race-free). Keyed by guest address in an open-addressed table. */
#if 1 /* lwmutex/lwcond: real on every host (win32_compat.h shims the Win32 sync API off Windows) */
#define LWM_HASH 65536u
static struct LwmSlot { volatile long addr; HANDLE sem;
    volatile long holder; volatile long long acq_us; volatile long long acq_fences;
    volatile unsigned long long acq_cpu_us;
    volatile long state; /* v96b fast lock word: 0 free, 1 held, 2 held + maybe waiters */
    volatile long waiters; /* v98b: threads blocked in lock / lwcond re-lock (fair handoff) */ } g_lwm[LWM_HASH];
extern "C" { extern volatile long long g_gcm_ref_pub_count;    /* cellGcmSys.c */
             unsigned long long ppu_thread_cpu_us(unsigned tid);   /* sys_ppu_thread.c */
             unsigned ppu_thread_prof_pc(unsigned tid); }
static volatile long g_lwm_tab_lock = 0;
/* PS3_LWMUTEX_TRACE=1: reconstruct the lwmutex lock-convoy that stalls LBP's loader.
 * Records, per mutex, the acquiring tid + a QPC microsecond timestamp; on a
 * contended block it logs who holds it and for how long; on unlock it flags a
 * long hold. The leaf holder (the one blocked on a non-lwmutex wait) is the
 * convoy root. Default OFF. */
static int lwm_trace(void){ static int v=-1; if(v<0){const char*e=getenv("PS3_LWMUTEX_TRACE"); v=e?1:0;} return v; }
static long long lwm_now_us(void){
#ifdef _WIN32
    static LARGE_INTEGER freq={0}; if(!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (long long)(c.QuadPart*1000000ll/freq.QuadPart);
#else
    return 0;
#endif
}
/* Find an EXISTING slot (no create) for hold-tracking. */
static struct LwmSlot* lwm_find(uint32_t addr){
    if(!addr) return nullptr;
    uint32_t h=(addr*2654435761u)&(LWM_HASH-1);
    for(uint32_t i=0;i<LWM_HASH;i++){ uint32_t idx=(h+i)&(LWM_HASH-1);
        long cur=g_lwm[idx].addr;
        if((uint32_t)cur==addr) return &g_lwm[idx];
        if(cur==0) return nullptr; }
    return nullptr;
}
static struct LwmSlot* lwm_slot(uint32_t addr);
static HANDLE lwm_sem(uint32_t addr)
{
    struct LwmSlot* sl = lwm_slot(addr);
    return sl ? sl->sem : nullptr;
}
static struct LwmSlot* lwm_slot(uint32_t addr)
{
    if (!addr) return nullptr;
    uint32_t h = (addr * 2654435761u) & (LWM_HASH - 1);
    for (uint32_t i = 0; i < LWM_HASH; i++) {
        uint32_t idx = (h + i) & (LWM_HASH - 1);
        long cur = g_lwm[idx].addr;
        if ((uint32_t)cur == addr) return &g_lwm[idx];
        if (cur == 0) {
            while (_InterlockedExchange(&g_lwm_tab_lock, 1)) YieldProcessor();
            struct LwmSlot* r = nullptr;
            if (g_lwm[idx].addr == 0) {
                g_lwm[idx].sem = CreateSemaphoreA(NULL, 1, 1, NULL); /* free */
                g_lwm[idx].state = 0;                                /* free (v96b fast lock) */
                g_lwm[idx].addr = (long)addr;   /* publish AFTER init (x86 TSO: readers see init) */
                r = &g_lwm[idx];
            } else if ((uint32_t)g_lwm[idx].addr == addr) {
                r = &g_lwm[idx];
            }
            _InterlockedExchange(&g_lwm_tab_lock, 0);
            if (r) return r;
            /* someone else claimed this slot for a different addr -> keep probing */
        }
    }
    return nullptr;   /* table full (raise LWM_HASH) */
}
/* v96b (FPS): user-space lwmutex fast path. The semaphore model cost two kernel calls
 * per uncontended lock/unlock pair (WaitForSingleObject(s, 0) + ReleaseSemaphore), ~3%
 * of the main thread in the city. Same semantics on a lock word (Drepper's futex mutex,
 * WaitOnAddress): no owner (any thread may release, like the semaphore), releasing a
 * free lock is a no-op (like the max-count-1 semaphore), timed waits honoured.
 * Chosen ONCE per run (both models must never mix on one mutex): PS3_LWM_FAST=0 keeps
 * the semaphores. Not switchable by PS3_AB for the same reason. */
#ifdef _WIN32
#  pragma comment(lib, "Synchronization.lib")
#endif
static int lwm_fast(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("PS3_LWM_FAST"); v = (e && *e == '0') ? 0 : 1;
                 fprintf(stderr, "[LWM] user-space lwmutex fast path: %s (v96b; PS3_LWM_FAST=0 = semaphores)\n",
                         v ? "ON" : "off"); }
    return v;
}
static bool lwm_word_trylock(struct LwmSlot* sl)
{
    return _InterlockedCompareExchange(&sl->state, 1, 0) == 0;
}
static bool lwm_word_lock(struct LwmSlot* sl, DWORD ms)   /* false = timed out */
{
    if (_InterlockedCompareExchange(&sl->state, 1, 0) == 0) return true;
    const ULONGLONG t0 = (ms == INFINITE) ? 0 : GetTickCount64();
    long c = _InterlockedExchange(&sl->state, 2);
    while (c != 0) {
        DWORD w = INFINITE;
        if (ms != INFINITE) {
            const ULONGLONG el = GetTickCount64() - t0;
            if (el >= ms) return false;
            w = (DWORD)(ms - el);
        }
        long cmp = 2;
        WaitOnAddress(&sl->state, &cmp, sizeof(long), w);
        c = _InterlockedExchange(&sl->state, 2);
    }
    return true;
}
static void lwm_word_unlock(struct LwmSlot* sl)
{
    if (_InterlockedExchange(&sl->state, 0) == 2) WakeByAddressSingle((PVOID)&sl->state);
}
/* v98b PS3_LWM_FAIR (default 1; 0 = v96b barging): lv2 hands a contended lwmutex to a sleeping waiter on
 * unlock. The v96b lock word let the unlocker take it straight back: the Linux smoke shows FIOS' HDD
 * mediathreads (tid 10-12, prio 99) blocked 2-4 s at a time (28 s of lwmutex waiting per 10 s) on
 * 0x405C59F8 while the fios scheduler (tid 13) unlocks and re-locks it in a loop -> streaming starves.
 * Now a locker that finds the word FREE while others sleep on it gives a woken waiter up to ~1-2 ms to take
 * it first (<= ~1-2 ms). Only the fast (word) model needs it: an NT semaphore release satisfies the waiter directly. */
static int lwm_fair(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("PS3_LWM_FAIR"); v = (e && *e == '0') ? 0 : 1;
                 fprintf(stderr, "[LWM] v98b fair lwmutex handoff: %s (PS3_LWM_FAIR=0 = v96b barging)\n", v ? "ON" : "off"); }
    return v;
}
static void lwm_yield_to_waiter(struct LwmSlot* sl)
{
    for (int i = 0; i < 256 + 192; i++) {   /* ~256 pauses, then up to 192 yields (~1-2 ms worst case) */
        if (sl->state != 0 || sl->waiters <= 0) return;   /* a waiter took it / nobody waits */
        if (i < 256) YieldProcessor(); else SwitchToThread();
    }
}
static void lwm_word_reset(struct LwmSlot* sl)   /* create at a reused address: force free */
{
    _InterlockedExchange(&sl->state, 0);
#ifdef _WIN32
    WakeByAddressAll((PVOID)&sl->state);
#else
    WakeByAddressSingle((PVOID)&sl->state);   /* the POSIX shim's wake is a broadcast */
#endif
}
#else /* unreachable */
static void* lwm_sem(uint32_t) { return nullptr; }
#endif
/* Contention-probe window flag: 0 by default (prints stay bounded). A title's
 * diagnostic code may set it around a suspect wait to uncap the [LWM-BLOCK]
 * logging during that window only (park hunts: gate on state, not counts). */
/* LWM_TRACE_EA=<hexEA>: log every lock (incl. recursive) and unlock of that
 * one mutex with the resulting recursion count and the caller's guest code
 * pointers -- pairs pushes with pops to find the lock whose release never
 * comes. Diagnostic only. */
static uint32_t lwm_trace_ea(void){ static uint32_t v=1; if(v==1){const char*e=getenv("LWM_TRACE_EA"); v=e?(uint32_t)strtoul(e,0,16):0;} return v; }
static void lwm_trace_log(ppu_context* ctx, uint32_t lwm, const char* what, uint32_t recur)
{
    uint32_t sp = (uint32_t)ctx->gpr[1];
    char b[600]; int p = snprintf(b, sizeof b, "[LWM-TR] tid=%llu %s lwm=0x%08X recur=%u code:",
                                  (unsigned long long)ctx->thread_id, what, lwm, recur);
    uint32_t prev = 0; int found = 0;
    for (uint32_t a = sp; a < sp + 0x800 && found < 8; a += 4) {
        uint32_t v = vm_read32(a);
        if (v >= 0x00010000u && v < 0x00700000u && !(v & 3) && v != prev) {
            p += snprintf(b + p, sizeof(b) - p, " %08X", v); prev = v; found++; }
    }
    fprintf(stderr, "%s\n", b);
}
volatile int g_nd_inpump = 0;
static void sys_lwmutex_lock(ppu_context* ctx)
{
    uint32_t lwm = (uint32_t)ctx->gpr[3];
    uint32_t self = LWM_SELF(ctx);
    /* lv2 ABI: r4 = timeout in microseconds, 0 = infinite. The real kernel
     * returns ETIMEDOUT (0x8001000B) when the wait expires; games rely on that
     * (e.g. LBP's resource loader locks with a 2s timeout in a retry loop so a
     * contended lock yields to other threads instead of hard-blocking). We had
     * been ignoring r4 and always waiting INFINITE, which defeats that pattern. */
    uint64_t timeout_us = ctx->gpr[4];
#if 1 /* lwmutex/lwcond: real on every host (win32_compat.h shims the Win32 sync API off Windows) */
    struct LwmSlot* sl = lwm_slot(lwm); HANDLE s = sl ? sl->sem : nullptr;
    if (s) {
        /* Recursive re-lock by the current holder: bump the count, no wait.
         * Only the holder ever stamps owner=self, so this check is race-free. */
        if (vm_read32(lwm + LWM_OWNER) == self && vm_read32(lwm + LWM_RECUR) > 0) {
            vm_write32(lwm + LWM_RECUR, vm_read32(lwm + LWM_RECUR) + 1);
            if (lwm_trace_ea() == lwm) lwm_trace_log(ctx, lwm, "relock", vm_read32(lwm + LWM_RECUR));
            ctx->gpr[3] = 0;
            return;
        }
        if (lwm_fast() && sl->waiters > 0 && lwm_fair()) lwm_yield_to_waiter(sl);   /* v98b */
        if (lwm_fast() ? !lwm_word_trylock(sl) : WaitForSingleObject(s, 0) != WAIT_OBJECT_0) {
            /* Contended: log who we're stuck behind (owner stamped at acquire),
             * then block. Bounded diagnostics for park hunts; uncapped while the
             * probe window is open (g_nd_inpump). */
            static long _bl = 0; long _b = ++_bl;
            if (g_nd_inpump || _b <= 40) fprintf(stderr, "[LWM-BLOCK] tid=%llu lwm=0x%08X owner=%u recur=%u tmo=%lluus\n",
                (unsigned long long)ctx->thread_id, lwm, vm_read32(lwm + LWM_OWNER), vm_read32(lwm + LWM_RECUR),
                (unsigned long long)timeout_us);
            if (lwm_trace()) { struct LwmSlot* sl = lwm_find(lwm);
                long h = sl ? sl->holder : 0; long long held = (sl && h) ? (lwm_now_us() - sl->acq_us) : 0;
                fprintf(stderr, "[LWM-CONVOY] tid=%llu BLOCKs lwm=0x%08X -> held-by-tid=%ld for %lldus (guest-owner=%u)\n",
                    (unsigned long long)ctx->thread_id, lwm, h, held, vm_read32(lwm + LWM_OWNER)); fflush(stderr); }
            long long _blk_start = lwm_trace() ? lwm_now_us() : 0;
            DWORD ms = INFINITE;
            if (timeout_us) { uint64_t m = (timeout_us + 999) / 1000; ms = m > 0xFFFFFFFEull ? 0xFFFFFFFEu : (DWORD)m; }
            InterlockedIncrement(&sl->waiters);   /* v98b */
            DWORD wr = lwm_fast() ? (lwm_word_lock(sl, ms) ? WAIT_OBJECT_0 : WAIT_TIMEOUT)
                                  : WaitForSingleObject(s, ms);
            InterlockedDecrement(&sl->waiters);
            if (wr == WAIT_TIMEOUT) {           /* honor the timeout: ETIMEDOUT, no acquire */
                if (lwm_trace()) fprintf(stderr, "[LWM-CONVOY] tid=%llu TIMED-OUT on lwm=0x%08X after %lldus (ETIMEDOUT, retry)\n",
                    (unsigned long long)ctx->thread_id, lwm, lwm_now_us() - _blk_start);
                ctx->gpr[3] = (uint64_t)(int64_t)(int32_t)0x8001000Bu;
                return;
            }
            if (lwm_trace()) fprintf(stderr, "[LWM-CONVOY] tid=%llu ACQUIRED lwm=0x%08X after waiting %lldus\n",
                (unsigned long long)ctx->thread_id, lwm, lwm_now_us() - _blk_start);
            if (g_nd_inpump || _b <= 40) fprintf(stderr, "[LWM-GOT] tid=%llu lwm=0x%08X\n", (unsigned long long)ctx->thread_id, lwm);
        }
    }
    if (lwm_trace()) { struct LwmSlot* sl = lwm_find(lwm); if (sl) { sl->holder = (long)self; sl->acq_us = lwm_now_us(); sl->acq_fences = g_gcm_ref_pub_count; sl->acq_cpu_us = ppu_thread_cpu_us(self); } }
#endif
    vm_write32(lwm + LWM_OWNER, self);
    vm_write32(lwm + LWM_RECUR, 1);
    if (lwm_trace_ea() == lwm) lwm_trace_log(ctx, lwm, "lock", 1);
    ctx->gpr[3] = 0;   // CELL_OK
}
static void sys_lwmutex_trylock(ppu_context* ctx)
{
    uint32_t lwm = (uint32_t)ctx->gpr[3];
    uint32_t self = LWM_SELF(ctx);
#if 1 /* lwmutex/lwcond: real on every host (win32_compat.h shims the Win32 sync API off Windows) */
    struct LwmSlot* sl = lwm_slot(lwm); HANDLE s = sl ? sl->sem : nullptr;
    if (s) {
        if (vm_read32(lwm + LWM_OWNER) == self && vm_read32(lwm + LWM_RECUR) > 0) {
            vm_write32(lwm + LWM_RECUR, vm_read32(lwm + LWM_RECUR) + 1);
            ctx->gpr[3] = 0;
            return;
        }
        if (lwm_fast() ? !lwm_word_trylock(sl) : WaitForSingleObject(s, 0) != WAIT_OBJECT_0) { ctx->gpr[3] = (uint64_t)(int64_t)(int32_t)0x8001000Bu; return; } // EBUSY
    }
    if (lwm_trace()) { struct LwmSlot* sl = lwm_find(lwm); if (sl) { sl->holder = (long)self; sl->acq_us = lwm_now_us(); sl->acq_fences = g_gcm_ref_pub_count; sl->acq_cpu_us = ppu_thread_cpu_us(self); } }
#endif
    vm_write32(lwm + LWM_OWNER, self);
    vm_write32(lwm + LWM_RECUR, 1);
    ctx->gpr[3] = 0;
}
static void sys_lwmutex_unlock(ppu_context* ctx)
{
    uint32_t lwm = (uint32_t)ctx->gpr[3];
    uint32_t rc = vm_read32(lwm + LWM_RECUR);
    if (lwm_trace_ea() == lwm) lwm_trace_log(ctx, lwm, "unlock", rc ? rc - 1 : 0);
    if (rc > 1) {                       /* recursive hold: count down, keep the lock */
        vm_write32(lwm + LWM_RECUR, rc - 1);
        ctx->gpr[3] = 0;
        return;
    }
    vm_write32(lwm + LWM_RECUR, 0);
    vm_write32(lwm + LWM_OWNER, 0);
#if 1 /* lwmutex/lwcond: real on every host (win32_compat.h shims the Win32 sync API off Windows) */
    if (lwm_trace()) { struct LwmSlot* sl = lwm_find(lwm);
        if (sl && sl->holder) { long long held = lwm_now_us() - sl->acq_us;
            if (held > 100000) {
                long long fences = g_gcm_ref_pub_count - sl->acq_fences;
                unsigned long long cpu_now = ppu_thread_cpu_us((unsigned)sl->holder);
                long long cpu_delta = (cpu_now && sl->acq_cpu_us) ? (long long)(cpu_now - sl->acq_cpu_us) : -1;
                fprintf(stderr, "[LWM-CONVOY] tid=%ld RELEASES lwm=0x%08X after holding %lldus (LONG HOLD) fences=%lld cpu=%lldus (%.0f%% cpu-bound) last-hle-from=0x%08X\n",
                    sl->holder, lwm, held, fences, cpu_delta,
                    cpu_delta >= 0 ? 100.0 * (double)cpu_delta / (double)held : -1.0,
                    ppu_thread_prof_pc((unsigned)sl->holder));
                /* Name the critical section: the unlocker IS the holder, so its
                 * guest stack right now is the exit of the long-held region.
                 * Back-chain LR slots are 0 under the DRAIN/fragment model, so
                 * scan the stack for words in the lifted code range instead
                 * (saved return addresses; a lifted func name IS its guest
                 * addr). Same idiom as [exit-chain] in ppu_hle.cpp. */
                uint32_t sp = (uint32_t)ctx->gpr[1];
                char b[1600]; int p = snprintf(b, sizeof b, "[LWM-CONVOY]   holder-bt sp=0x%08X codeptrs:", sp);
                uint32_t prev = 0; int found = 0;
                for (uint32_t a = sp; a < sp + 0x3000 && found < 48; a += 4) {
                    uint32_t v = vm_read32(a);
                    if (v >= 0x00010000u && v < 0x00900000u && v != prev) {
                        p += snprintf(b + p, sizeof(b) - p, " %08X", v); prev = v; found++; }
                }
                fprintf(stderr, "%s\n", b); fflush(stderr);
            }
            sl->holder = 0; } }
    struct LwmSlot* sl = lwm_slot(lwm); HANDLE s = sl ? sl->sem : nullptr;
    /* Semaphore release works from ANY thread (unlike a CS) -- guest code
     * hands lwmutex ownership across threads. Over-release (unlock of a free
     * mutex) fails harmlessly at the max count of 1. */
    if (s) { if (lwm_fast()) lwm_word_unlock(sl); else ReleaseSemaphore(s, 1, NULL); }
#endif
    ctx->gpr[3] = 0;
}

/* sys_lwcond (sysPrxForUser) — guest-side condition variable, paired with an
 * lwmutex. The CRT and (newly) libsre's cellSpurs create/wait/signal these. Like
 * sys_lwmutex above, model it directly in guest memory so the args stay GUEST
 * EAs (the generic adapter would pass them raw and the C sysPrxForUser impl
 * deref'd them as host pointers -> AV during cellSpurs init). A no-op wait is
 * adequate here: the CRT/SPURS paths that reach us use these for one-shot init
 * handshakes, not long-term blocking. PS3 guest pointers are 32-bit, so the
 * ABI layout is exactly 8 bytes: +0x00 lwmutex EA (be32), +0x04 queue id.
 * Writing a be64 pointer here used to leave +0x00 zero and overrun the object,
 * causing SDK wrappers to report every condition variable as invalid. */
static void sys_lwcond_create(ppu_context* ctx)
{
    static uint32_t s_lwcond_id = 0x4C000000u;
    static unsigned s_lwcond_create_count = 0;
    uint32_t lwcond  = (uint32_t)ctx->gpr[3];
    uint32_t lwmutex = (uint32_t)ctx->gpr[4];
    vm_write32(lwcond + 0x00, lwmutex);
    vm_write32(lwcond + 0x04, ++s_lwcond_id);
    if (s_lwcond_create_count++ < 32) {
        fprintf(stderr,
                "[LWCOND] create cond=0x%08X mutex=0x%08X queue=0x%08X tid=%llu lr=0x%08X\n",
                lwcond, lwmutex, s_lwcond_id,
                (unsigned long long)ctx->thread_id, (uint32_t)ctx->lr);
    }
    ctx->gpr[3] = 0;
}
/* REAL lwcond (2026-10-03). The old model -- signal = no-op, wait = release,
 * Sleep(1), reacquire -- is only correct for callers that re-check their
 * predicate in a loop. inFamous' SP Movie Player does NOT: func_0018621C sets
 * player+0x48 = 1 ("need texture pool"), waits ONCE, then proceeds if +0x48 is
 * still non-zero. With a 1 ms spurious wake it always proceeded before the main
 * thread had handed over the pool (-> the movie-pool race / stall), and its
 * "done, return the pool" handshake (+0x48 = 3) broke the same way.
 *
 * Now: waiters block until signal/signal_all/signal_to, or the guest timeout
 * (r4, usec, 0 = infinite -> ETIMEDOUT 0x8001000B like lv2). Safety net: an
 * untimed wait still wakes after PS3_LWCOND_MAX_MS (default 3000, 0 = never)
 * and logs it, so an HLE path that wakes guests some other way degrades to the
 * old poll behaviour instead of a hard hang. Strict waiting applies only to the
 * call sites selected by lwc_strict() below; everything else keeps the old poll. */
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <chrono>
#include <atomic>
namespace {
struct LwcState { int waiters = 0; int pending = 0; };
/* Old model (LWCOND_PER_OBJ=0, PS3_AB phase LWCOND_AB): ONE host mutex + ONE condvar for every lwcond,
 * so every signal notify_all'ed every waiter of every lwcond in the process. */
std::mutex g_lwc_m;
std::condition_variable g_lwc_cv;
std::unordered_map<uint32_t, LwcState> g_lwc;
/* v103C per-object model (LWCOND_PER_OBJ=1, default): the same waiters/pending bookkeeping, but kept
 * in one of LWC_NB buckets selected by the lwcond EA, each with its own mutex + condvar. A signal only
 * wakes the waiters of its bucket (in practice: of its own lwcond). Semantics are unchanged: register
 * as a waiter before dropping the lwmutex, pending <= waiters, non-strict waits still return after
 * the poll cap (LWCOND_POLL_MS, default 1 ms = the old behaviour FIOS relies on), strict waits as before. */
enum { LWC_NB = 256 };
struct LwcBucket { std::mutex m; std::condition_variable cv; std::unordered_map<uint32_t, LwcState> map; };
LwcBucket g_lwcb[LWC_NB];
inline LwcBucket& lwc_bucket(uint32_t cond) { return g_lwcb[(uint32_t)((cond >> 3) * 2654435761u) >> 24]; }
std::atomic<long> g_lwc_old_waiters{0};   /* waiters currently parked in the OLD structure (phase switches) */
/* [LWCOND-STAT] counters (every 10 s) */
std::atomic<long long> s_lws_wait{0}, s_lws_strict{0}, s_lws_sig_wake{0}, s_lws_tmo_wake{0}, s_lws_poll_wake{0},
    s_lws_wait_us{0}, s_lws_sig{0}, s_lws_sig_nowaiter{0}, s_lws_relock_us{0}, s_lws_old{0};
std::atomic<long long> s_lws_t0{0};
struct LwsSite { std::atomic<uint32_t> lr{0}; std::atomic<long long> waits{0}, sig{0}, poll{0}; };
LwsSite s_lws_site[16];
bool lws_stats_on()
{
    static const bool on = [] {
        const char* e = getenv("LWCOND_STAT");
        return !(e && *e == '0');
    }();
    return on;
}
int lwc_per_obj_env() { static int v = -1; if (v < 0) { const char* e = getenv("LWCOND_PER_OBJ"); v = (e && *e == '0') ? 0 : 1; } return v; }
int lwc_ab_phase() { static int v = -2; if (v == -2) { const char* e = getenv("LWCOND_AB"); v = e ? atoi(e) : 5; } return v; }
int lwc_ab_long_phase() { static int v = -2; if (v == -2) { const char* e = getenv("LWCOND_AB_LONG"); v = e ? atoi(e) : 6; } return v; }
}
extern "C" volatile int g_ab_mask;
namespace {
bool lwc_per_obj() { return lwc_per_obj_env() && g_ab_mask != lwc_ab_phase(); }
long long lwc_poll_us()
{
    static long long v = -1, vl = -1;
    if (v < 0) { const char* e = getenv("LWCOND_POLL_MS"); double ms = e ? atof(e) : 1.0; if (ms < 0.05) ms = 0.05; v = (long long)(ms * 1000.0);
                 const char* e2 = getenv("LWCOND_POLL_LONG_MS"); double ml = e2 ? atof(e2) : 20.0; if (ml < 0.05) ml = 0.05; vl = (long long)(ml * 1000.0);
                 fprintf(stderr, "[LWCOND] v103C per-lwcond waits: %s (LWCOND_PER_OBJ; PS3_AB phase %d = old global condvar), non-strict poll cap %.2f ms "
                                 "(LWCOND_POLL_MS; PS3_AB phase %d = %.2f ms LWCOND_POLL_LONG_MS)\n",
                         lwc_per_obj_env() ? "ON" : "off", lwc_ab_phase(), v / 1000.0, lwc_ab_long_phase(), vl / 1000.0); }
    return (lwc_per_obj_env() && g_ab_mask == lwc_ab_long_phase()) ? vl : v;
}
long long lwc_now_us()
{
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}
LwsSite* lws_site(uint32_t lr)
{
    for (int i = 0; i < 16; i++) {
        uint32_t cur = s_lws_site[i].lr.load(std::memory_order_relaxed);
        if (cur == lr) return &s_lws_site[i];
        if (cur == 0) { uint32_t z = 0; if (s_lws_site[i].lr.compare_exchange_strong(z, lr) || z == lr) return &s_lws_site[i]; }
    }
    return nullptr;
}
void lws_report()
{
    if (!lws_stats_on()) return;
    long long now = lwc_now_us(), t0 = s_lws_t0.load();
    if (t0 == 0) { s_lws_t0.compare_exchange_strong(t0, now); return; }
    if (now - t0 < 10000000 || !s_lws_t0.compare_exchange_strong(t0, now)) return;
    long long w = s_lws_wait.exchange(0), st = s_lws_strict.exchange(0), sw = s_lws_sig_wake.exchange(0),
              tw = s_lws_tmo_wake.exchange(0), pw = s_lws_poll_wake.exchange(0), us = s_lws_wait_us.exchange(0),
              sg = s_lws_sig.exchange(0), sn = s_lws_sig_nowaiter.exchange(0), rl = s_lws_relock_us.exchange(0), od = s_lws_old.exchange(0);
    char b[900]; int p = snprintf(b, sizeof b,
        "[LWCOND-STAT] 10s: %lld waits (%lld strict, %lld in old model), signalled wakes %lld, guest-timeout wakes %lld, poll/fallback wakes %lld, "
        "avg wait %.0f us, lwmutex re-lock avg %.1f us; %lld signals (%lld with no waiter); per-obj %s poll %.2f ms (ab %d) | top sites:",
        w, st, od, sw, tw, pw, w ? (double)us / w : 0.0, w ? (double)rl / w : 0.0, sg, sn, lwc_per_obj() ? "ON" : "off", lwc_poll_us() / 1000.0, g_ab_mask);
    for (int i = 0; i < 16 && p < (int)sizeof b - 60; i++) {
        uint32_t lr = s_lws_site[i].lr.load(); if (!lr) break;
        long long sw2 = s_lws_site[i].waits.exchange(0), ss = s_lws_site[i].sig.exchange(0), sp = s_lws_site[i].poll.exchange(0);
        if (sw2) p += snprintf(b + p, sizeof b - p, " lr %08X %lld w/%lld sig/%lld poll", lr, sw2, ss, sp);
    }
    fprintf(stderr, "%s\n", b);
}
/* Which waits block for real. Default: NONE (old poll model everywhere), because
 * other code (FIOS' timed waits, at least) still depends on the 1 ms wake and
 * stalled completely when every wait was strict (2026-10-03 Windows run).
 * PS3_LWCOND_STRICT_LR=<hex lr>[,...]: strict only for waits called from these
 *   guest return addresses (inFamous sets 18631C,18638C = SP Movie Player).
 * PS3_LWCOND_STRICT=1: strict for every wait (debug; currently stalls FIOS).
 * PS3_LWCOND_POLL=1: never strict. */
bool lwc_strict(uint32_t lr)
{
    static int all = -1, never = -1; static uint32_t lrs[16]; static int nlr = -1;
    if (all < 0) { const char* e = getenv("PS3_LWCOND_STRICT"); all = (e && *e && *e != '0') ? 1 : 0; }
    if (never < 0) { const char* e = getenv("PS3_LWCOND_POLL"); never = (e && *e && *e != '0') ? 1 : 0; }
    if (nlr < 0) {
        nlr = 0;
        if (const char* e = getenv("PS3_LWCOND_STRICT_LR")) {
            const char* q = e;
            while (*q && nlr < 16) { char* end = nullptr; unsigned long v = strtoul(q, &end, 16);
                if (end == q) break; lrs[nlr++] = (uint32_t)v; q = end; while (*q == ',' || *q == ' ') q++; }
            fprintf(stderr, "[LWCOND] strict waits at %d call site(s) (PS3_LWCOND_STRICT_LR=%s)\n", nlr, e);
        }
    }
    if (never) return false;
    if (all) return true;
    for (int i = 0; i < nlr; i++) if (lrs[i] == lr) return true;
    return false;
}
/* Some statically linked middleware uses a lwcond as a one-shot handoff and
 * assumes the producer cannot run past the consumer's wait setup.  Host
 * scheduling can violate that assumption: a signal then lands just before the
 * waiter is registered and ordinary condition-variable semantics discard it.
 * Keep one pending credit only for explicitly selected guest lwcond objects;
 * all other lwconds retain normal non-sticky semantics. */
bool lwc_sticky(uint32_t cond)
{
    static uint32_t conds[16]; static int ncond = -1;
    if (ncond < 0) {
        ncond = 0;
        if (const char* e = getenv("PS3_LWCOND_STICKY")) {
            const char* q = e;
            while (*q && ncond < 16) {
                char* end = nullptr; unsigned long v = strtoul(q, &end, 16);
                if (end == q) break;
                conds[ncond++] = (uint32_t)v;
                q = end; while (*q == ',' || *q == ' ') q++;
            }
            fprintf(stderr, "[LWCOND] sticky wake credit for %d object(s) (PS3_LWCOND_STICKY=%s)\n", ncond, e);
        }
    }
    for (int i = 0; i < ncond; i++) if (conds[i] == cond) return true;
    return false;
}
long lwc_max_ms() { static long v = -2; if (v == -2) { const char* e = getenv("PS3_LWCOND_MAX_MS"); v = e ? atol(e) : 3000; } return v; }
int lwc_trace() { static int v = -1; if (v < 0) { const char* e = getenv("PS3_LWCOND_TRACE"); v = e ? atoi(e) : 0; } return v; }
void lwc_wake(uint32_t cond, bool all, ppu_context* ctx, const char* what)
{
    int w = 0;
    const bool stats = lws_stats_on();
    if (stats) s_lws_sig++;
    {   /* v103C per-object structure */
        LwcBucket& b = lwc_bucket(cond);
        std::lock_guard<std::mutex> g(b.m);
        auto it = b.map.find(cond);
        int bw = (it == b.map.end()) ? 0 : it->second.waiters;
        if (bw > 0) {
            if (all) it->second.pending = bw;
            else if (it->second.pending < bw) it->second.pending++;
            b.cv.notify_all();
        } else if (lwc_sticky(cond)) {
            /* Collapse any number of early signals into one one-shot credit. */
            b.map[cond].pending = 1;
        }
        w += bw;
    }
    /* old structure: in the old model (LWCOND_PER_OBJ=0 / its A/B phase) or while waiters that
     * started in it are still parked there (phase switch) */
    if (!lwc_per_obj() || g_lwc_old_waiters.load() > 0) {
        std::lock_guard<std::mutex> g(g_lwc_m);
        auto it = g_lwc.find(cond);
        int ow = (it == g_lwc.end()) ? 0 : it->second.waiters;
        if (ow > 0) {
            if (all) it->second.pending = ow;
            else if (it->second.pending < ow) it->second.pending++;
            g_lwc_cv.notify_all();
        } else if (lwc_sticky(cond)) {
            g_lwc[cond].pending = 1;
        }
        w += ow;
    }
    if (stats && w == 0) s_lws_sig_nowaiter++;
    static long n = 0;
    if (lwc_trace() && n++ < lwc_trace())
        fprintf(stderr, "[LWCOND] %s cond=0x%08X waiters=%d tid=%llu lr=0x%08X\n", what, cond, w,
                (unsigned long long)ctx->thread_id, (uint32_t)ctx->lr);
}
} /* namespace */
static void sys_lwcond_destroy(ppu_context* ctx)
{
    uint32_t cond = (uint32_t)ctx->gpr[3];
    { std::lock_guard<std::mutex> g(g_lwc_m); auto it = g_lwc.find(cond); if (it != g_lwc.end() && it->second.waiters == 0) g_lwc.erase(it); }
    { LwcBucket& b = lwc_bucket(cond); std::lock_guard<std::mutex> g(b.m);
      auto it = b.map.find(cond); if (it != b.map.end() && it->second.waiters == 0) b.map.erase(it); }
    ctx->gpr[3] = 0;
}
static void sys_lwcond_signal(ppu_context* ctx)     { lwc_wake((uint32_t)ctx->gpr[3], false, ctx, "signal");     ctx->gpr[3] = 0; }
static void sys_lwcond_signal_all(ppu_context* ctx) { lwc_wake((uint32_t)ctx->gpr[3], true,  ctx, "signal_all"); ctx->gpr[3] = 0; }
static void sys_lwcond_signal_to(ppu_context* ctx)  { lwc_wake((uint32_t)ctx->gpr[3], false, ctx, "signal_to");  ctx->gpr[3] = 0; }
/* Release the paired lwmutex, block until signalled (see above), reacquire.
 * Registers as a waiter BEFORE dropping the guest mutex so a signal sent by the
 * next holder cannot be lost. Handles recursive holds by saving the count.
 * v103C: the waiter bookkeeping lives in the lwcond's bucket (per-object) unless LWCOND_PER_OBJ=0 or
 * the PS3_AB phase LWCOND_AB (default 5) selects the old global structure. */
static void sys_lwcond_wait(ppu_context* ctx)
{
    uint32_t lwcond  = (uint32_t)ctx->gpr[3];
    uint32_t lwmutex = vm_read32(lwcond + 0x00);
    uint64_t timeout_us = ctx->gpr[4];
    int rc_ret = 0;
#if 1 /* lwmutex/lwcond: real on every host (win32_compat.h shims the Win32 sync API off Windows) */
    struct LwmSlot* sl = lwm_slot(lwmutex); HANDLE s = sl ? sl->sem : nullptr;
    if (s) {
        uint32_t own = vm_read32(lwmutex + LWM_OWNER);
        uint32_t rc  = vm_read32(lwmutex + LWM_RECUR);
        const bool stats = lws_stats_on();
        if (stats) lws_report();
        const long long t_w0 = stats ? lwc_now_us() : 0;
        LwsSite* site = stats ? lws_site((uint32_t)ctx->lr) : nullptr;
        if (site) site->waits++;
        if (stats) s_lws_wait++;
        /* v28 (FPS): the non-strict "poll" wait used a blind Sleep(1) -- a signalled
         * waiter still slept its full 1-2 ms, and every signal->wake hop on the
         * frame's critical path paid it. Now it waits on the same condvar the
         * strict path uses, with a 1 ms cap: an unsignalled wait behaves exactly
         * as before (wake after ~1 ms and re-check, which FIOS relies on), a
         * signalled one wakes immediately. PS3_LWCOND_POLL_SLEEP=1 -> old Sleep(1). */
        static int s_poll_sleep = -1;
        if (s_poll_sleep < 0) { const char* e = getenv("PS3_LWCOND_POLL_SLEEP"); s_poll_sleep = (e && *e == '1') ? 1 : 0; }
        bool strict = lwc_strict((uint32_t)ctx->lr);
        const bool use_obj = lwc_per_obj();
        LwcBucket& bk = lwc_bucket(lwcond);
        std::mutex& M = use_obj ? bk.m : g_lwc_m;
        std::condition_variable& CV = use_obj ? bk.cv : g_lwc_cv;
        std::unordered_map<uint32_t, LwcState>& MAP = use_obj ? bk.map : g_lwc;
        if (!use_obj && stats) s_lws_old++;
        if (!strict && s_poll_sleep) {
            vm_write32(lwmutex + LWM_RECUR, 0);
            vm_write32(lwmutex + LWM_OWNER, 0);
            if (lwm_fast()) lwm_word_unlock(sl); else ReleaseSemaphore(s, 1, NULL);
            Sleep(1);
            if (stats) s_lws_poll_wake++; if (site) site->poll++;
        } else if (!strict) {
            if (!use_obj) g_lwc_old_waiters++;
            std::unique_lock<std::mutex> lk(M);
            MAP[lwcond].waiters++;
            vm_write32(lwmutex + LWM_RECUR, 0);
            vm_write32(lwmutex + LWM_OWNER, 0);
            if (lwm_fast()) lwm_word_unlock(sl); else ReleaseSemaphore(s, 1, NULL);
            long long cap_us = use_obj ? lwc_poll_us() : 1000;
            const bool guest_cap = timeout_us && (long long)timeout_us < cap_us;
            if (guest_cap) cap_us = (long long)timeout_us;
            bool got = CV.wait_for(lk, std::chrono::microseconds(cap_us),
                                   [&]{ return MAP[lwcond].pending > 0; });
            LwcState& st = MAP[lwcond];
            if (got) st.pending--;
            st.waiters--;
            if (st.pending > st.waiters) st.pending = st.waiters;
            lk.unlock();
            if (!use_obj) g_lwc_old_waiters--;
            if (got) { if (stats) s_lws_sig_wake++; if (site) site->sig++; }
            else if (guest_cap) { if (stats) s_lws_tmo_wake++; }
            else { if (stats) s_lws_poll_wake++; if (site) site->poll++; }
        } else {
            if (stats) s_lws_strict++;
            if (!use_obj) g_lwc_old_waiters++;
            std::unique_lock<std::mutex> lk(M);
            MAP[lwcond].waiters++;
            vm_write32(lwmutex + LWM_RECUR, 0);
            vm_write32(lwmutex + LWM_OWNER, 0);
            if (lwm_fast()) lwm_word_unlock(sl); else ReleaseSemaphore(s, 1, NULL);
            long max_ms = lwc_max_ms();
            bool guest_tmo = timeout_us != 0;
            long long lim_ms = guest_tmo ? (long long)((timeout_us + 999) / 1000) : (long long)max_ms;
            auto pred = [&]{ return MAP[lwcond].pending > 0; };
            bool got;
            if (!guest_tmo && max_ms <= 0) { CV.wait(lk, pred); got = true; }
            else got = CV.wait_for(lk, std::chrono::milliseconds(lim_ms), pred);
            LwcState& st = MAP[lwcond];
            if (got) st.pending--;
            st.waiters--;
            if (st.pending > st.waiters) st.pending = st.waiters;
            lk.unlock();
            if (!use_obj) g_lwc_old_waiters--;
            if (got) { if (stats) s_lws_sig_wake++; if (site) site->sig++; }
            else if (guest_tmo) { if (stats) s_lws_tmo_wake++; }
            else { if (stats) s_lws_poll_wake++; if (site) site->poll++; }
            if (!got) {
                if (guest_tmo) rc_ret = (int)0x8001000Bu;   /* ETIMEDOUT */
                else {
                    static long nto = 0;
                    if (nto++ < 40)
                        fprintf(stderr, "[LWCOND] wait cond=0x%08X tid=%llu lr=0x%08X: no signal in %ld ms -> waking anyway (safety)\n",
                                lwcond, (unsigned long long)ctx->thread_id, (uint32_t)ctx->lr, max_ms);
                }
            }
        }
        const long long t_w1 = stats ? lwc_now_us() : 0;
        if (stats) s_lws_wait_us += t_w1 - t_w0;
        if (lwm_fast() && sl->waiters > 0 && lwm_fair()) lwm_yield_to_waiter(sl);   /* v98b */
        InterlockedIncrement(&sl->waiters);
        if (lwm_fast()) lwm_word_lock(sl, INFINITE); else WaitForSingleObject(s, INFINITE);
        InterlockedDecrement(&sl->waiters);
        if (stats) s_lws_relock_us += lwc_now_us() - t_w1;
        vm_write32(lwmutex + LWM_OWNER, own);
        vm_write32(lwmutex + LWM_RECUR, rc ? rc : 1);
    }
#endif
    ctx->gpr[3] = (uint64_t)(int64_t)(int32_t)rc_ret;
}

/* sys_ppu_thread_get_id(vm::ptr<u64> id) -> *id = calling thread's real id.
 * The old fixed "1" broke every am-I-the-designated-thread check in
 * multithreaded titles (LBP's job system routes work by thread identity, so
 * its queues were never serviced and network-init parked forever). */
static void sys_ppu_thread_get_id(ppu_context* ctx)
{
    uint32_t p = (uint32_t)ctx->gpr[3];
    /* Every registered thread has a nonzero id (main = 1 via
     * ppu_thread_register_main); 0 = unregistered scratch ctx, report the same
     * never-a-real-thread sentinel the lwmutex owner stamps use. */
    if (p) vm_write64(p, ctx->thread_id ? (uint64_t)ctx->thread_id : 0x7FFFFFFEull);
    ctx->gpr[3] = 0;
}

/* sys_mmapper_allocate_memory(u32 size, u64 flags, vm::ptr<u32> mem_id) ->
 * hand back a unique opaque id; the backing is the flat VM, so the later
 * search_and_map just needs a non-zero id to track. */
/* id -> size so sys_mmapper_search_and_map (lv2 337) can lay blocks out
 * without overlap. Ids are dense from 0x1000. */
static uint32_t s_mm_sizes[256];
static uint32_t s_mmapper_next_id = 0x1000;
extern "C" uint32_t ps3_mmapper_block_size(uint32_t mem_id)
{
    uint32_t i = mem_id - 0x1000u;
    return (i < 256) ? s_mm_sizes[i] : 0;
}

static uint32_t mmapper_new_id(uint32_t size)
{
    uint32_t id = s_mmapper_next_id++;
    if (id - 0x1000u < 256) s_mm_sizes[id - 0x1000u] = size;
    return id;
}

static void sys_mmapper_allocate_memory(ppu_context* ctx)
{
    uint32_t size       = (uint32_t)ctx->gpr[3];
    uint32_t mem_id_ptr = (uint32_t)ctx->gpr[5];
    uint32_t id         = mmapper_new_id(size);
    if (getenv("PS3_MEMTRACE"))
        fprintf(stderr, "[mmapper] allocate_memory(size=0x%X flags=0x%llX id_ptr=0x%X) -> id 0x%X\n",
                size, (unsigned long long)ctx->gpr[4], mem_id_ptr, id);
    if (mem_id_ptr) vm_write32(mem_id_ptr, id);
    ctx->gpr[3] = 0;
}
/* sys_mmapper_allocate_memory_from_container(u32 size, u32 container, u64 flags,
 * vm::ptr<u32> mem_id) -> id in *r6. flОw's CRT uses this for its heap/mutex pool;
 * it was previously UNregistered (CRT saw failure -> "not enough memory"). */
static void sys_mmapper_allocate_memory_from_container(ppu_context* ctx)
{
    uint32_t size = (uint32_t)ctx->gpr[3];
    uint32_t mem_id_ptr = (uint32_t)ctx->gpr[6];
    uint32_t id = mmapper_new_id(size);
    if (getenv("PS3_MEMTRACE"))
        fprintf(stderr, "[mmapper] alloc_from_container(size=0x%X cid=0x%X flags=0x%llX id_ptr=0x%X) -> id 0x%X\n",
                size, (uint32_t)ctx->gpr[4], (unsigned long long)ctx->gpr[5], mem_id_ptr, id);
    if (mem_id_ptr) vm_write32(mem_id_ptr, id);
    ctx->gpr[3] = 0;
}

/* A handful of CRT helpers the early boot tends to hit; accept and continue. */
static void crt_ok(ppu_context* ctx) { ctx->gpr[3] = 0; }
static void sys_lwmutex_destroy_counted(ppu_context* ctx)
{
    { static long long _n=0; _n++;
      if (getenv("LWM_COUNT") && (_n<=24 || (_n%50000)==0))
        fprintf(stderr, "[LWM] destroy #%lld lwm=0x%08X r4=0x%08X r5=0x%08X\n", _n,
                (uint32_t)ctx->gpr[3], (uint32_t)ctx->gpr[4], (uint32_t)ctx->gpr[5]); }
    ctx->gpr[3] = 0;
}

/* Real preemptive thread create/exit live in the lv2 syscall layer
 * (syscalls/sys_ppu_thread.c) and spawn a host thread that runs the guest
 * entry through the recompiled code. The CRT also reaches them as
 * sysPrxForUser import NIDs (gen_hle_nids can't see them — they're not defined
 * in the sysPrxForUser lib), so bridge the NIDs to the same implementation.
 * Without this the CRT's thread/static-init runs through an uninitialised
 * object table and calls heap addresses as function pointers. */
extern "C" int64_t sys_ppu_thread_create(ppu_context* ctx);
extern "C" int64_t sys_ppu_thread_exit(ppu_context* ctx);
/* The ctx-aware dispatch (ppu_hle.cpp) does NOT propagate a handler return value
 * into gpr[3] -- each ctx handler must set gpr[3] itself. sys_ppu_thread_create
 * signals success by *returning* CELL_OK(0) (it never writes gpr[3]), so we must
 * store that return into gpr[3]. Otherwise gpr[3] is left as the incoming out-ptr
 * (&tid, nonzero) and the guest wrapper reads it as "create failed" -- e.g. LBP's
 * sub_52613C does `v6 = (ret==0); return v6 ? tid : 0`, so a nonzero ret makes it
 * hand back 0 and the caller's init (sub_C1484 / KdConvert) bails. */
static void hle_ppu_thread_create(ppu_context* ctx) { ctx->gpr[3] = sys_ppu_thread_create(ctx); }
static void hle_ppu_thread_exit(ppu_context* ctx)   { ctx->gpr[3] = sys_ppu_thread_exit(ctx); }

/* _cellGcmInitBody (NID 0x15BAE46B) -- the GCM init every PS3 game calls via the
 * cellGcmInit() SDK macro. cellGcmSys.c provides the layout-correct core
 * (cellGcmSetupContext) but needs the owning vm to allocate the guest
 * CellGcmContextData and write the game's context-out pointer; supply those as
 * callbacks. Without this the game's GCM context stays null -> null deref. */
typedef unsigned int (*CellGcmGuestAlloc)(unsigned int, unsigned int);
typedef void (*CellGcmGuestWrite32)(unsigned int, unsigned int);
extern "C" unsigned int cellGcmSetupContext(unsigned int ctx_out_addr,
    unsigned int cmdSize, unsigned int ioSize, unsigned int ioAddress,
    CellGcmGuestAlloc galloc, CellGcmGuestWrite32 gwrite32);

static unsigned int gcm_guest_alloc(unsigned int size, unsigned int align)
{
    /* Bump from a small scratch region below the main stack (0x0FF00000) and
     * above the TLS image -- a few control structs, never freed. */
    static unsigned int bump = 0x0F800000u;
    if (align < 16) align = 16;
    bump = (bump + align - 1) & ~(align - 1);
    unsigned int a = bump;
    bump += (size + 15u) & ~15u;
    return a;
}
static void gcm_guest_write32(unsigned int addr, unsigned int val) { vm_write32(addr, val); }

/* FIFO command-buffer-full callback. cellGcmSetupContext points the guest
 * context's callback OPD at GCM_FIFO_CALLBACK_SENTINEL_EA; the title's inline
 * gcmReserve calls context->callback(context, count) on ring wrap, which the
 * indirect dispatcher routes here. r3 = guest context EA.
 *
 * Derived from VM_HLE_INJECT_BASE, the same base libs/video/cellGcmSys.c
 * builds the OPD from. This was a second hardcoded copy of the address, and
 * moving the injected block out of guest-allocatable memory updated only one
 * of them: the OPD then pointed at an EA with no registered function, the
 * garbage-vcall guard no-opped the callback, the FIFO never recycled and the
 * title stopped flipping. One definition, one place. */
#define GCM_FIFO_CALLBACK_SENTINEL_EA (VM_HLE_INJECT_BASE + 0x2F00u)
extern "C" void cellGcm_fifo_recycle(unsigned int ctx_ea);
extern "C" void ppu_register_function(uint64_t addr, void (*fn)(ppu_context*));
static void hle_gcm_callback(ppu_context* ctx)
{
    cellGcm_fifo_recycle((unsigned int)ctx->gpr[3]);   /* r3 = context EA */
    ctx->gpr[3] = 0;                                   /* CELL_OK */
}

static void hle_cellGcmInitBody(ppu_context* ctx)
{
    uint32_t ctx_out = (uint32_t)ctx->gpr[3];
    uint32_t cmdSize = (uint32_t)ctx->gpr[4];
    uint32_t ioSize  = (uint32_t)ctx->gpr[5];
    uint32_t ioAddr  = (uint32_t)ctx->gpr[6];
    fprintf(stderr, "[HLE] _cellGcmInitBody(ctx_out=0x%08X, cmdSize=0x%X, ioSize=0x%X, ioAddr=0x%X)\n",
            ctx_out, cmdSize, ioSize, ioAddr);
    cellGcmSetupContext(ctx_out, cmdSize, ioSize, ioAddr, gcm_guest_alloc, gcm_guest_write32);
    ctx->gpr[3] = 0;   /* CELL_OK */
}

/* --- sys_net offline model ---------------------------------------------
 * LBP's net-services tick (sub_11A864) drains its UDP socket with
 * non-blocking recvfrom until it returns -1 (empty socket = EWOULDBLOCK on
 * real firmware). These NIDs were unresolved, and the unresolved default of
 * r3=0 reads as "received a 0-byte packet": the drain loop spins forever
 * while holding the net-manager lwmutex, wedging the whole boot at the
 * Network init node. Model the offline truth instead: no data, no sockets --
 * every receive/poll would-block. errno lives in a guest scratch cell since
 * _sys_net_errno_loc returns a POINTER the game dereferences. */
#define SYS_NET_EWOULDBLOCK_V 35
extern "C" void ps3_net_host_register(unsigned int (*guest_alloc)(unsigned int, unsigned int));
static uint32_t g_net_errno_ea = 0;
static void hle_net_errno_loc(ppu_context* ctx)
{
    if (!g_net_errno_ea) g_net_errno_ea = gcm_guest_alloc(4, 4);
    vm_write32(g_net_errno_ea, SYS_NET_EWOULDBLOCK_V);
    ctx->gpr[3] = g_net_errno_ea;
}
static void hle_net_wouldblock(ppu_context* ctx)
{
    if (g_net_errno_ea) vm_write32(g_net_errno_ea, SYS_NET_EWOULDBLOCK_V);
    ctx->gpr[3] = (uint64_t)(int64_t)(int32_t)-1;
}
static void hle_net_zero(ppu_context* ctx) { ctx->gpr[3] = 0; }
/* select/poll: nothing is ever ready offline -- but a real select BLOCKS for
 * the caller's timeout before saying so. Returning instantly turned LBP's
 * 30Hz net pump (sub_3A5548: select(1, r/w/e sets, {0s, 33333us})) into a
 * 100%-CPU busy-spin that also dominated the guest-PC profiler, masquerading
 * as a boot hang. Honor the timeout and clear the fd sets (1024-bit each). */
static void hle_net_select(ppu_context* ctx)
{
    uint32_t rd = (uint32_t)ctx->gpr[4], wr = (uint32_t)ctx->gpr[5];
    uint32_t ex = (uint32_t)ctx->gpr[6], tv = (uint32_t)ctx->gpr[7];
    uint64_t us = 10000;              /* NULL timeout = block forever: tick at 10ms instead */
    if (tv) us = vm_read64(tv) * 1000000ull + vm_read64(tv + 8);   /* {s64 sec, s64 usec} BE */
    if (us > 100000) us = 100000;     /* cap so shutdown stays responsive */
    if (us) Sleep((DWORD)((us + 999) / 1000));
    const uint32_t sets[3] = { rd, wr, ex };
    for (int s = 0; s < 3; s++)
        if (sets[s]) for (uint32_t i = 0; i < 128; i += 4) vm_write32(sets[s] + i, 0);
    ctx->gpr[3] = 0;                  /* 0 fds ready */
}
static void hle_net_poll(ppu_context* ctx)
{
    uint32_t fds  = (uint32_t)ctx->gpr[3];
    uint32_t nfds = (uint32_t)ctx->gpr[4];
    int32_t  ms   = (int32_t)(uint32_t)ctx->gpr[5];
    if (ms < 0 || ms > 100) ms = (ms < 0) ? 10 : 100;   /* -1 = infinite: tick at 10ms */
    if (ms) Sleep((DWORD)ms);
    for (uint32_t i = 0; i < nfds && i < 64; i++)       /* pollfd = {s32 fd, s16 ev, s16 rev} */
        if (fds) vm_write32(fds + i * 8 + 4, vm_read32(fds + i * 8 + 4) & 0xFFFF0000u);
    ctx->gpr[3] = 0;                  /* 0 fds ready */
}
/* Distinct small fds: the unresolved default handed EVERY socket() call fd 0,
 * making all sockets alias one id in the game's tables. */
static void hle_net_socket(ppu_context* ctx) { static uint32_t s_fd = 3; ctx->gpr[3] = s_fd++; }
/* sendto: report the full length as sent (packets vanish into the void, matching
 * the RPCS3-offline oracle where broadcasts go out and nothing answers). */
static void hle_net_sendto(ppu_context* ctx) { ctx->gpr[3] = (uint32_t)ctx->gpr[5]; }

/* _sys_spu_image_import (sysPrxForUser NID 0xEBE5F72F) -- the user-space wrapper
 * libsre calls during cellSpursInitialize to load the SPURS SPU kernel into the
 * sys_spu_image struct. Previously UNRESOLVED -> returned 0 -> libsre proceeded
 * with an EMPTY image (entry=0) -> the 5 SPURS SPU threads ran nothing -> PPU
 * busy-waits on an SPU completion that never comes -> 0 GCM draws
 * (prx/spu_kernel/README.txt step 1). Parses the SPU ELF at r4 into the image
 * struct at r3.
 *
 * The ELF-parse (segment/entry layout below) is lifted directly from sagemono's
 * sys_spu_image_import SYSCALL handler in PR #57 (fix/spu-image-import); this is
 * the sysPrxForUser LIBRARY counterpart libsre actually calls. Credit: sagemono
 * (PR #57) for the import implementation + SPU-image syscall-number fix.
 * Self-verifying: no-op unless r4 points to an ELF (so a wrong NID guess is safe).
 * sys_spu_image { u32 type; u32 entry; sys_spu_segment* segs; int nsegs; }
 * sys_spu_segment{ int type; u32 ls_start; int size; u64 src_pa }  (0x18, src@0x10) */
/* Last SPU image parsed below: where its loadable bytes live, which local
 * store address they start at, and how far they span. Read by the SPU DMA
 * path (SPU_DSP_IMAGE_EA in spu_dma.h) to recover a section load whose
 * source address the title lost between import and use. */
extern "C" uint32_t g_spu_image_src_ea = 0, g_spu_image_ls_start = 0,
                    g_spu_image_span = 0;

/* img_ea -> the EA of the SPU ELF it was parsed from. Small and fixed: a title
 * has a handful of images, and a repeat import of the same descriptor replaces
 * its entry. */
#define SPU_IMG_SRC_MAX 32
static struct { uint32_t img, src; } s_spu_img_src[SPU_IMG_SRC_MAX];

extern "C" void ps3_spu_image_record(uint32_t img_ea, uint32_t src_ea)
{
    if (!img_ea || !src_ea) return;
    for (int i = 0; i < SPU_IMG_SRC_MAX; i++)
        if (s_spu_img_src[i].img == img_ea || s_spu_img_src[i].img == 0) {
            s_spu_img_src[i].img = img_ea; s_spu_img_src[i].src = src_ea; return;
        }
}

extern "C" uint32_t ps3_spu_image_source_ea(uint32_t img_ea)
{
    for (int i = 0; i < SPU_IMG_SRC_MAX; i++)
        if (s_spu_img_src[i].img == img_ea) return s_spu_img_src[i].src;
    return 0;
}

extern "C" void spu_raw_note_image(uint32_t src_ea, uint32_t entry);  /* runtime/spu/spu_raw.c */

static void hle_sys_spu_image_import(ppu_context* ctx)
{
    uint32_t img_ea = (uint32_t)ctx->gpr[3];
    uint32_t src_ea = (uint32_t)ctx->gpr[4];
    fprintf(stderr, "[HLE] _sys_spu_image_import(img=0x%08X src=0x%08X r5=0x%08X r6=0x%08X)\n",
            img_ea, src_ea, (uint32_t)ctx->gpr[5], (uint32_t)ctx->gpr[6]);
    if (!img_ea || !src_ea || !vm_base) { ctx->gpr[3] = 0; return; }
    const uint8_t* e = vm_base + src_ea;
    if (!(e[0]==0x7F && e[1]=='E' && e[2]=='L' && e[3]=='F')) {
        fprintf(stderr, "[HLE] _sys_spu_image_import: src not an ELF -> no-op\n");
        fflush(stderr); ctx->gpr[3] = 0; return;
    }
    uint16_t machine = (uint16_t)((e[0x12] << 8) | e[0x13]);   /* 23 = SPU */
    uint32_t entry   = vm_read32(src_ea + 0x18);
    uint32_t phoff   = vm_read32(src_ea + 0x1C);
    uint16_t phentsz = (uint16_t)((e[0x2A] << 8) | e[0x2B]); if (!phentsz) phentsz = 0x20;
    uint16_t phnum   = (uint16_t)((e[0x2C] << 8) | e[0x2D]);
    static uint32_t s_seg_bump = 0x0D000000u;
    uint32_t segs_ea = s_seg_bump; int nsegs = 0;
    for (uint16_t i = 0; i < phnum && nsegs < 32; i++) {
        uint32_t ph = phoff + (uint32_t)i * phentsz;
        if (vm_read32(src_ea + ph + 0x00) != 1) continue;      /* PT_LOAD */
        uint32_t p_off = vm_read32(src_ea + ph + 0x04);
        uint32_t p_va  = vm_read32(src_ea + ph + 0x08);
        uint32_t p_fsz = vm_read32(src_ea + ph + 0x10);
        uint32_t p_msz = vm_read32(src_ea + ph + 0x14);
        uint32_t seg = segs_ea + (uint32_t)nsegs * 0x18;       /* COPY */
        vm_write32(seg + 0x00, 1); vm_write32(seg + 0x04, p_va);
        vm_write32(seg + 0x08, p_fsz);
        /* src is the u64 at +0x10; its low word, which every 32-bit reader
         * takes, is +0x14 -- but LBP's FMOD mixer reads the u32 at +0x10 as
         * the DMA source (it DMA'd its DSP overlay from EA 0 and jumped into
         * zeroed LS), so write both, as the lv2 syscall path already does. */
        vm_write32(seg + 0x10, src_ea + p_off);
        vm_write32(seg + 0x14, src_ea + p_off); nsegs++;
        if (p_msz > p_fsz && nsegs < 32) {                     /* BSS tail -> FILL 0 */
            seg = segs_ea + (uint32_t)nsegs * 0x18;
            vm_write32(seg + 0x00, 2); vm_write32(seg + 0x04, p_va + p_fsz);
            vm_write32(seg + 0x08, p_msz - p_fsz);
            vm_write32(seg + 0x10, 0); vm_write32(seg + 0x14, 0); nsegs++;
        }
    }
    s_seg_bump += (uint32_t)nsegs * 0x18;
    if (s_seg_bump >= 0x0E000000u) s_seg_bump = 0x0D000000u;
    /* Remember where this image's loadable bytes live, for the SPU-side
     * recovery in spu_dma.h (SPU_DSP_IMAGE_EA). The segments are contiguous in
     * the source image, so segment 0's address plus the span from its ls_start
     * to the end of the last segment describes the whole thing. */
    if (nsegs > 0) {
        uint32_t last = segs_ea + (uint32_t)(nsegs - 1) * 0x18;
        g_spu_image_src_ea   = vm_read32(segs_ea + 0x14);
        g_spu_image_ls_start = vm_read32(segs_ea + 0x04);
        g_spu_image_span     = vm_read32(last + 0x04) - g_spu_image_ls_start;
    }
    /* Remember which ELF this descriptor was parsed from. sys_spu_thread_group_
     * start only gets the DESCRIPTOR, but the workload registry is keyed by a
     * fingerprint of the ELF's own bytes -- without this the raw SPU path has no
     * way to ask whether the image it is about to run was lifted. */
    ps3_spu_image_record(img_ea, src_ea);
    /* ...and ask that question now, while the ELF is still identified. A raw SPU
     * is started by an MMIO store to its run-control register, with no syscall in
     * between, so this import is the last point at which the image can be matched
     * to a lifted entry (runtime/spu/spu_raw.c). No-op for a SPU-thread image. */
    spu_raw_note_image(src_ea, entry);
    vm_write32(img_ea + 0x00, 0);                              /* type = USER */
    vm_write32(img_ea + 0x04, entry);
    vm_write32(img_ea + 0x08, nsegs ? segs_ea : 0);
    vm_write32(img_ea + 0x0C, (uint32_t)nsegs);
    fprintf(stderr, "[HLE] _sys_spu_image_import -> entry=0x%05X nsegs=%d machine=%u (SPU=23)\n",
            entry, nsegs, machine);
    /* SPU_IMAGE_DIAG=1: r6 is a caller-supplied guest buffer that differs on
     * every call, which this handler ignores -- it points the segments at the
     * source image instead. Show the segments written and what the caller's
     * buffer holds, to establish whether the caller expects it to be filled. */
    if (getenv("SPU_IMAGE_DIAG")) {
        for (int i = 0; i < nsegs; i++) {
            uint32_t s = segs_ea + (uint32_t)i * 0x18;
            fprintf(stderr, "        seg[%d] type=%u ls=0x%05X size=%u src=0x%08X%08X\n",
                    i, vm_read32(s + 0x00), vm_read32(s + 0x04), vm_read32(s + 0x08),
                    vm_read32(s + 0x10), vm_read32(s + 0x14));
        }
        uint32_t r6 = (uint32_t)ctx->gpr[6];
        if (r6 && r6 < 0xD0000000u) {
            /* Scan a window around the caller's buffer for the DSP descriptor
             * the SPU later reads: its first word is 0x0052E1E8 and it carries
             * the image's {entry, size, ls_start, size} at +0x150. Finding it
             * at a fixed offset says r6 IS that descriptor. */
            for (int32_t o = -0x400; o <= 0x400; o += 4) {
                uint32_t a = (uint32_t)((int32_t)r6 + o);
                if (vm_read32(a) == 0x0052E1E8u)
                    fprintf(stderr, "        descriptor marker 0x0052E1E8 at r6%+d (0x%08X)\n", o, a);
                if (vm_read32(a) == 0x00000248u && vm_read32(a + 4) == 0x00002CB0u)
                    fprintf(stderr, "        image quad {248,2CB0,80,2CB0} at r6%+d (0x%08X) "
                            "=> descriptor+0x150, descriptor=0x%08X, its +0x140=%08X %08X %08X %08X\n",
                            o, a, a - 0x150,
                            vm_read32(a - 0x10), vm_read32(a - 0xC),
                            vm_read32(a - 8), vm_read32(a - 4));
            }
        }
    }
    fflush(stderr);
    ctx->gpr[3] = 0;
}

/* ---- sys_spinlock_* (sysPrxForUser) -------------------------------------
 *
 * A sys_spinlock_t is one 32-bit word in guest memory and nothing else -- the
 * real firmware spins on it with lwarx/stwcx. The guest only ever touches that
 * word through these four calls, so a host test-and-set on the same address is
 * exactly equivalent and stays correct across the title's own threads.
 *
 * Stored big-endian, so a guest that peeks at the word still reads 1.
 *
 * ABI: initialize / lock / unlock return void; trylock returns CELL_OK or
 * EBUSY. Found via Virtua Fighter 5, which locks with trylock in a retry loop
 * and was the first title to import the family -- the three NIDs were unnamed
 * in the database until compute_nid() was brute-forced over the sysPrxForUser
 * export list.
 */
#define SPINLOCK_BE1  0x01000000u          /* be32(1) */

static inline volatile long* spin_word(uint32_t ea)
{
    return (volatile long*)(vm_base + ea);
}

/* Test-and-set, not compare-and-swap: writing "locked" over an already-locked
 * word is idempotent, so a plain atomic exchange is enough -- and _Interlocked-
 * Exchange is the one RMW the POSIX shim in win32_compat.h already provides. */
static inline int spin_try(uint32_t ea)
{
    return _InterlockedExchange(spin_word(ea), (long)SPINLOCK_BE1) == 0;
}

static void sys_spinlock_initialize(ppu_context* ctx)
{
    uint32_t ea = (uint32_t)ctx->gpr[3];
    if (ea) _InterlockedExchange(spin_word(ea), 0);
}

/* Who currently holds each locked word, so a spin that never ends can name the
 * holder instead of just burning a core in silence.
 *
 * A spinlock has no timeout and no failure return, so a lock that is never
 * released is invisible: the thread sits in Sleep(0) at 100% of a core, a
 * sampling profiler attributes every sample to ntdll, and NOTHING says which
 * address or which holder. Guitar Hero III deadlocks exactly this way during
 * its heap bring-up and presented as "all threads idle in ntdll" while pinning
 * a core -- two facts that only make sense together once the lock is named.
 *
 * Small fixed table, linear scan: contention here is rare by construction (a
 * guest spinlock section is a few instructions), and a miss just means the
 * report says "unknown" rather than being wrong. */
#define SPIN_OWNER_MAX 64
static struct { uint32_t ea; unsigned long tid; } s_spin_owner[SPIN_OWNER_MAX];

static void spin_note_acquire(uint32_t ea)
{
    for (int i = 0; i < SPIN_OWNER_MAX; i++)
        if (s_spin_owner[i].ea == 0 || s_spin_owner[i].ea == ea) {
            s_spin_owner[i].ea = ea; s_spin_owner[i].tid = GetCurrentThreadId(); return;
        }
}
static void spin_note_release(uint32_t ea)
{
    for (int i = 0; i < SPIN_OWNER_MAX; i++)
        if (s_spin_owner[i].ea == ea) { s_spin_owner[i].ea = 0; s_spin_owner[i].tid = 0; return; }
}
static unsigned long spin_owner_of(uint32_t ea)
{
    for (int i = 0; i < SPIN_OWNER_MAX; i++)
        if (s_spin_owner[i].ea == ea) return s_spin_owner[i].tid;
    return 0;
}

static void sys_spinlock_lock(ppu_context* ctx)
{
    uint32_t ea = (uint32_t)ctx->gpr[3];
    if (!ea) return;
    /* ponytail: spin briefly, then hand the core back. On real hardware the
     * holder runs to completion in a few instructions; here it is a host thread
     * the OS can deschedule mid-section, so busy-waiting a whole quantum is the
     * wrong trade. Swap in a futex if a title ever shows real contention here. */
    for (unsigned n = 0; !spin_try(ea); n++) {
        if (n < 64) YieldProcessor();
        else        Sleep(0);
        /* Report once per stuck lock, well past any legitimate section. */
        if (n == 200000) {
            unsigned long owner = spin_owner_of(ea);
            unsigned long me    = GetCurrentThreadId();
            static int reported = 0;
            if (reported++ < 8)
                fprintf(stderr,
                        "[spinlock] STUCK on 0x%08X after 200k spins: guest lr=0x%08X, "
                        "held by tid %lu, this is tid %lu%s\n",
                        ea, (uint32_t)ctx->lr, owner, me,
                        (owner && owner == me)
                          ? "  <== SELF-DEADLOCK: this thread already holds it"
                          : (owner ? "" : "  (holder unknown -- released without us seeing it?)"));
        }
    }
    spin_note_acquire(ea);
}

static void sys_spinlock_trylock(ppu_context* ctx)
{
    uint32_t ea = (uint32_t)ctx->gpr[3];
    int got = (ea && spin_try(ea));
    if (got) spin_note_acquire(ea);
    ctx->gpr[3] = got
                ? 0                                            /* CELL_OK */
                : (uint64_t)(int64_t)(int32_t)0x8001000Au;     /* EBUSY   */
}

static void sys_spinlock_unlock(ppu_context* ctx)
{
    uint32_t ea = (uint32_t)ctx->gpr[3];
    if (!ea) return;
    spin_note_release(ea);
    _InterlockedExchange(spin_word(ea), 0);
}

extern "C" void ppu_sysprx_register(void)
{
    ps3_hle_register_ctx(0x15BAE46Bu, "_cellGcmInitBody", hle_cellGcmInitBody);
    ps3_hle_register_ctx(0xEBE5F72Fu, "_sys_spu_image_import", hle_sys_spu_image_import);

    /* PS3_NET_ONLINE: real host sockets (libs/network/sysNet.c). Registered
     * first because the first registration of a NID wins, so these shadow the
     * offline model below; without the variable nothing changes. */
    if (getenv("PS3_NET_ONLINE")) ps3_net_host_register(gcm_guest_alloc);
    /* sys_net offline model (NIDs from PSL1GHT libnet exports). Covers every
     * sys_net NID LBP imports so none fall to the unresolved-NID default. */
    ps3_hle_register_ctx(0x6005CDE1u, "_sys_net_errno_loc",     hle_net_errno_loc);
    ps3_hle_register_ctx(0x1F953B9Fu, "sys_net_bnet_recvfrom",  hle_net_wouldblock);
    ps3_hle_register_ctx(0xFBA04F37u, "sys_net_bnet_recv",      hle_net_wouldblock);
    ps3_hle_register_ctx(0xC9D09C34u, "sys_net_bnet_recvmsg",   hle_net_wouldblock);
    ps3_hle_register_ctx(0x051EE3EEu, "sys_net_bnet_poll",      hle_net_poll);
    ps3_hle_register_ctx(0x3F09E20Au, "sys_net_bnet_select",    hle_net_select);
    ps3_hle_register_ctx(0x139A9E9Bu, "netInitializeNetworkEx", hle_net_zero);   /* lib init ok */
    ps3_hle_register_ctx(0x9C056962u, "netSocket",              hle_net_socket);
    ps3_hle_register_ctx(0xB0A59804u, "netBind",                hle_net_zero);
    ps3_hle_register_ctx(0x88F03575u, "netSetSockOpt",          hle_net_zero);
    ps3_hle_register_ctx(0x9647570Bu, "netSendTo",              hle_net_sendto);
    ps3_hle_register_ctx(0x6DB6E8CDu, "netClose",               hle_net_zero);
    ps3_hle_register_ctx(0x71F4C717u, "netGetHostByName",       hle_net_zero);   /* NULL: DNS down */
    ps3_hle_register_ctx(0xB68D5625u, "netFinalizeNetwork",     hle_net_zero);
    ps3_hle_register_ctx(0xFDB8F926u, "netFreethreadContext",   hle_net_zero);
    /* Route the GCM command-buffer-full callback (invoked indirectly via the
     * context OPD) into cellGcm_fifo_recycle so the FIFO ring recycles on wrap. */
    ppu_register_function(GCM_FIFO_CALLBACK_SENTINEL_EA, hle_gcm_callback);
    ps3_hle_register_ctx(ps3_compute_nid("sys_initialize_tls"),       "sys_initialize_tls",       sys_initialize_tls);
    ps3_hle_register_ctx(ps3_compute_nid("sys_time_get_system_time"), "sys_time_get_system_time", sys_time_get_system_time);
    ps3_hle_register_ctx(ps3_compute_nid("sys_process_is_stack"),     "sys_process_is_stack",     sys_process_is_stack);
    /* Atexit registration: nothing to do at boot, just succeed. */
    ps3_hle_register_ctx(ps3_compute_nid("_sys_process_atexitspawn"), "_sys_process_atexitspawn", crt_ok);
    ps3_hle_register_ctx(ps3_compute_nid("_sys_process_at_Exitspawn"),"_sys_process_at_Exitspawn",crt_ok);

    /* Lightweight mutex family (guards global/singleton init in the CRT). */
    ps3_hle_register_ctx(ps3_compute_nid("sys_lwmutex_create"),  "sys_lwmutex_create",  sys_lwmutex_create);
    ps3_hle_register_ctx(ps3_compute_nid("sys_lwmutex_destroy"), "sys_lwmutex_destroy", sys_lwmutex_destroy_counted);
    ps3_hle_register_ctx(ps3_compute_nid("sys_lwmutex_lock"),    "sys_lwmutex_lock",    sys_lwmutex_lock);
    ps3_hle_register_ctx(ps3_compute_nid("sys_lwmutex_unlock"),  "sys_lwmutex_unlock",  sys_lwmutex_unlock);
    ps3_hle_register_ctx(ps3_compute_nid("sys_lwmutex_trylock"), "sys_lwmutex_trylock", sys_lwmutex_trylock);

    /* Spinlocks. One guest word each, no kernel object; see the block above. */
    ps3_hle_register_ctx(ps3_compute_nid("sys_spinlock_initialize"), "sys_spinlock_initialize", sys_spinlock_initialize);
    ps3_hle_register_ctx(ps3_compute_nid("sys_spinlock_lock"),       "sys_spinlock_lock",       sys_spinlock_lock);
    ps3_hle_register_ctx(ps3_compute_nid("sys_spinlock_trylock"),    "sys_spinlock_trylock",    sys_spinlock_trylock);
    ps3_hle_register_ctx(ps3_compute_nid("sys_spinlock_unlock"),     "sys_spinlock_unlock",     sys_spinlock_unlock);

    ps3_hle_register_ctx(ps3_compute_nid("sys_lwcond_create"),     "sys_lwcond_create",     sys_lwcond_create);
    ps3_hle_register_ctx(ps3_compute_nid("sys_lwcond_destroy"),    "sys_lwcond_destroy",    sys_lwcond_destroy);
    ps3_hle_register_ctx(ps3_compute_nid("sys_lwcond_signal"),     "sys_lwcond_signal",     sys_lwcond_signal);
    ps3_hle_register_ctx(ps3_compute_nid("sys_lwcond_signal_all"), "sys_lwcond_signal_all", sys_lwcond_signal_all);
    ps3_hle_register_ctx(ps3_compute_nid("sys_lwcond_signal_to"),  "sys_lwcond_signal_to",  sys_lwcond_signal_to);
    ps3_hle_register_ctx(ps3_compute_nid("sys_lwcond_wait"),       "sys_lwcond_wait",       sys_lwcond_wait);

    /* Thread id + memory manager (high-frequency boot imports). The flat VM
     * means map/unmap/free are no-ops: the memory already exists everywhere. */
    ps3_hle_register_ctx(ps3_compute_nid("sys_ppu_thread_get_id"),      "sys_ppu_thread_get_id",      sys_ppu_thread_get_id);
    ps3_hle_register_ctx(ps3_compute_nid("sys_ppu_thread_create"),      "sys_ppu_thread_create",      hle_ppu_thread_create);
    ps3_hle_register_ctx(ps3_compute_nid("sys_ppu_thread_exit"),        "sys_ppu_thread_exit",        hle_ppu_thread_exit);
    ps3_hle_register_ctx(ps3_compute_nid("sys_mmapper_allocate_memory"), "sys_mmapper_allocate_memory", sys_mmapper_allocate_memory);
    ps3_hle_register_ctx(ps3_compute_nid("sys_mmapper_allocate_memory_from_container"), "sys_mmapper_allocate_memory_from_container", sys_mmapper_allocate_memory_from_container);
    ps3_hle_register_ctx(ps3_compute_nid("sys_mmapper_map_memory"),     "sys_mmapper_map_memory",     crt_ok);
    ps3_hle_register_ctx(ps3_compute_nid("sys_mmapper_unmap_memory"),   "sys_mmapper_unmap_memory",   crt_ok);
    ps3_hle_register_ctx(ps3_compute_nid("sys_mmapper_free_memory"),    "sys_mmapper_free_memory",    crt_ok);
}
