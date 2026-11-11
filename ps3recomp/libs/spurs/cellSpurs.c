/*
 * ps3recomp - cellSpurs HLE implementation
 *
 * Provides the SPURS management API so games can call SPU task/workload
 * functions without crashing.  Full SPU execution requires recompiling
 * SPU programs; this layer provides the scheduling and management APIs.
 *
 * Tasks and workloads are tracked.  If a game provides PPU fallback
 * callbacks, those can be invoked through the task submission path.
 */

#include "../video/rsx_getenv_cache.h"   /* cached getenv(): SPU hot paths called getenv per PM run (2026-10-04) */
#include "cellSpurs.h"
#include "../../runtime/platform/win32_compat.h"
#include "../../runtime/ps3_log.h"
#include "spu_workload.h"   /* SPU image -> lifted-entry dispatch (runtime/spu) */
#include "spurs_taskset.h"  /* REAL BE CellSpursTaskset layout builders (fork Option-B) */
#include "../../runtime/ppu/ppu_memory.h"   /* vm_base (guest mem) */
#include <stdio.h>
#include <stdlib.h>   /* getenv -- an implicit decl returns int, truncating the pointer */
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#ifndef SPU_PM_TLS
#ifdef _MSC_VER
#define SPU_PM_TLS __declspec(thread)
#else
#define SPU_PM_TLS _Thread_local
#endif
#endif

/* Bridge the real (BE) taskset EA + selected taskId from CreateTask to the image-22
 * SPU dispatch (spu_workload.c), so spurs_pm_build_context can build the leaf's
 * SpursTasksetContext from the real taskset. Set right before dispatch_async (the PPU
 * create path is sequential here). Gated by YDKJ_REAL_TASKSET in the dispatch. */
uint32_t g_ydkj_real_taskset_ea = 0;
uint32_t g_ydkj_real_taskid     = 0;
uint32_t g_ydkj_real_spurs_ea   = 0;   /* real CellSpurs instance EA (for the taskset-policy handoff) */

/* Generic HLE adapter passes GUEST addresses; translate pointer args. CellSpurs
 * is treated opaquely by the game (passed back as a handle), so translating the
 * pointer is enough here. */
#define GUEST_PTR(p, T) ((T)((p) ? (void*)(vm_base + (uint32_t)(uintptr_t)(p)) : (void*)0))

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#endif

/* ---------------------------------------------------------------------------
 * Internal workload tracking
 * -----------------------------------------------------------------------*/

typedef struct {
    int         in_use;
    const void* pm;            /* policy-module image EA (guest) */
    u32         sizePm;
    u64         data;          /* workload data (e.g. joblist EA) */
    u32         spurs_ea;      /* owning CellSpurs instance EA */
    u8          priority[CELL_SPURS_MAX_SPU];
    u32         minContention;
    u32         maxContention;
    u32         readyCount;
    int         started;       /* its SPU program has been dispatched        */
} SpursWorkload;

typedef struct {
    int         in_use;
    u32         id;
    int         active;
    int         completed;
    s32         exitCode;
    void*       entryPoint;
    u64         argA;
} SpursTask;

/* Global workload table (per-SPURS instance in a real system, simplified) */
static SpursWorkload s_workloads[CELL_SPURS_MAX_WORKLOAD];
static SpursTask     s_tasks[CELL_SPURS_MAX_TASK];
static u32           s_next_task_id = 0;

/* ---------------------------------------------------------------------------
 * Event flag sync side table
 *
 * The CellSpursEventFlag itself is 128 bytes of GAME-owned guest memory in
 * the REAL big-endian kernel layout (see EF_* offsets below) — the SPU side
 * (lifted task-library code) reads/writes it with DMA + atomics, so the PPU
 * HLE must operate on the same guest bytes, never a host struct. This side
 * table (keyed by guest EA) only adds the host mutex + condvar used to block
 * and wake PPU waiters.
 * -----------------------------------------------------------------------*/
#define MAX_EVENT_FLAGS 64

typedef struct {
    uint32_t            ea;      /* guest EA of the 128-byte flag */
#ifdef _WIN32
    CRITICAL_SECTION    cs;
    CONDITION_VARIABLE  cv;
#else
    pthread_mutex_t     mtx;
    pthread_cond_t      cond;
#endif
    int                 initialized;
} EventFlagSync;

static EventFlagSync s_ef_sync[MAX_EVENT_FLAGS];
/* v115 FTL request timeline (defined further down, next to FRAMEREC) */
static uint32_t ftl_line(void);
extern volatile uint32_t g_ftl_line;
void ps3_ftl(int type, uint32_t a, uint32_t b);
void ps3_fiosreq_woke(uint32_t flag, u16 bits);   /* v118 */

/* PPU threads inside cellSpursEventFlagWait. The SPU sets a flag with
 * GETLLAR/PUTLLC on the guest struct, which never touched the waiter's
 * condition variable, so every wait ran out its 2 ms poll: FMOD's mixer waits
 * on its SPU task once or twice per 5.3 ms audio block, fell behind, and
 * GH3's songs played ~60% silence (heard as static). */
static volatile long s_ef_ppu_waiters;
static void ef_line_committed(uint32_t line)
{
    if (!s_ef_ppu_waiters) return;
    for (int i = 0; i < MAX_EVENT_FLAGS; i++) {
        EventFlagSync* s = &s_ef_sync[i];
        if (!s->initialized || (s->ea & ~127u) != line) continue;
        /* Take the waiter's lock so the wake cannot fall between its check of
         * the flag and its sleep. Caller holds no lock-line lock. */
#ifdef _WIN32
        EnterCriticalSection(&s->cs); WakeAllConditionVariable(&s->cv); LeaveCriticalSection(&s->cs);
#else
        pthread_mutex_lock(&s->mtx); pthread_cond_broadcast(&s->cond); pthread_mutex_unlock(&s->mtx);
#endif
    }
}

static EventFlagSync* ef_sync_find(uint32_t ea)
{
    for (int i = 0; i < MAX_EVENT_FLAGS; i++) {
        if (s_ef_sync[i].initialized && s_ef_sync[i].ea == ea)
            return &s_ef_sync[i];
    }
    return NULL;
}

static EventFlagSync* ef_sync_alloc(uint32_t ea)
{
    for (int i = 0; i < MAX_EVENT_FLAGS; i++) {
        if (!s_ef_sync[i].initialized) {
            s_ef_sync[i].ea = ea;
            { extern void (*g_spu_line_commit_hook)(uint32_t);
              g_spu_line_commit_hook = ef_line_committed; }
#ifdef _WIN32
            InitializeCriticalSection(&s_ef_sync[i].cs);
            InitializeConditionVariable(&s_ef_sync[i].cv);
#else
            pthread_mutex_init(&s_ef_sync[i].mtx, NULL);
            pthread_cond_init(&s_ef_sync[i].cond, NULL);
#endif
            s_ef_sync[i].initialized = 1;
            return &s_ef_sync[i];
        }
    }
    return NULL;
}

/* Lenient lookup: a flag touched before we saw its Initialize (alternate init
 * paths, e.g. taskset2) still gets a sync slot instead of an error. */
static EventFlagSync* ef_sync_get(uint32_t ea)
{
    EventFlagSync* s = ef_sync_find(ea);
    if (!s) {
        s = ef_sync_alloc(ea);
        if (s)
            fprintf(stderr, "[cellSpurs] event flag 0x%08X used before Initialize "
                            "-- sync slot auto-allocated\n", ea);
    }
    return s;
}

static void ef_sync_free(EventFlagSync* sync)
{
    if (!sync) return;
#ifdef _WIN32
    DeleteCriticalSection(&sync->cs);
    /* CONDITION_VARIABLE has no destroy on Windows */
#else
    pthread_mutex_destroy(&sync->mtx);
    pthread_cond_destroy(&sync->cond);
#endif
    sync->ea = 0;
    sync->initialized = 0;
}

static inline void ef_lock(EventFlagSync* s)
{
#ifdef _WIN32
    EnterCriticalSection(&s->cs);
#else
    pthread_mutex_lock(&s->mtx);
#endif
}

static inline void ef_unlock(EventFlagSync* s)
{
#ifdef _WIN32
    LeaveCriticalSection(&s->cs);
#else
    pthread_mutex_unlock(&s->mtx);
#endif
}

/* Returns 1 if signaled, 0 if it timed out after `ms`. */
static inline int ef_wait_timed(EventFlagSync* s, unsigned ms)
{
#ifdef _WIN32
    return SleepConditionVariableCS(&s->cv, &s->cs, ms) ? 1 : 0;
#else
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += ms / 1000; ts.tv_nsec += (long)(ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    return pthread_cond_timedwait(&s->cond, &s->mtx, &ts) == 0 ? 1 : 0;
#endif
}

static inline void ef_broadcast(EventFlagSync* s)
{
#ifdef _WIN32
    WakeAllConditionVariable(&s->cv);
#else
    pthread_cond_broadcast(&s->cond);
#endif
}

/* ---------------------------------------------------------------------------
 * REAL CellSpursEventFlag guest layout (128 bytes, big-endian; RPCS3
 * cellSpurs.h is the reference). The SPU task library manipulates this exact
 * layout with GETLLAR/PUTLLC, so every PPU-side access goes through vm_*.
 * -----------------------------------------------------------------------*/
#define EF_EVENTS            0x00u  /* be u16: the event bits */
#define EF_SPU_PENDING_RECV  0x02u  /* be u16: slot bits with met conditions */
#define EF_PPU_WAIT_MASK     0x04u  /* be u16: blocked PPU thread's mask */
#define EF_PPU_WAIT_SLOTMODE 0x06u  /* u8: hi4 = wait slot, lo4 = wait mode */
#define EF_PPU_PENDING_RECV  0x07u  /* u8: 1 when the PPU waiter's cond met */
#define EF_SPU_USED_SLOTS    0x08u  /* be u16: wait slots in use (bit 0x8000>>s) */
#define EF_SPU_WAIT_MODE     0x0Au  /* be u16: per-slot mode (1 = AND) */
#define EF_SPU_PORT          0x0Cu  /* u8 */
#define EF_IS_IWL            0x0Du  /* u8: addr is a wkl instead of taskset */
#define EF_DIRECTION         0x0Eu  /* u8: CELL_SPURS_EVENT_FLAG_* direction */
#define EF_CLEAR_MODE        0x0Fu  /* u8: 0 = AUTO, 1 = MANUAL */
#define EF_SPU_WAIT_MASK_ARR 0x10u  /* be u16 [16]: per-slot wait masks */
#define EF_PENDING_RECV_EVT  0x30u  /* be u16 [16]: events handed to waiters */
#define EF_WAITING_TASK_ID   0x50u  /* u8 [16]: waiting task ids */
#define EF_WAITING_WKL_ID    0x60u  /* u8 [16]: waiting workload ids */
#define EF_ADDR              0x70u  /* be u64: taskset EA (isIwl=0) */
#define EF_EVENT_PORT_ID     0x78u  /* be u32 */
#define EF_EVENT_QUEUE_ID    0x7Cu  /* be u32 */
#define EF_GUEST_SIZE        0x80u

/* Layer-2 hook: wake an SPU task that registered a wait slot and slept via
 * the taskset syscall. Implemented in runtime/spu (task park/wake); the
 * event-flag core only reports WHO must wake. */
extern void spu_taskset_signal_task(uint32_t taskset_ea, uint32_t taskId);

/* Core Set protocol on the guest struct (sync lock held). Mirrors the real
 * kernel/RPCS3 semantics: satisfy registered SPU-task wait slots (slot s uses
 * bit 0x8000>>s in the used/pending/mode words), hand each its received
 * events, auto-clear consumed bits, then signal the tasks. PPU waiters are
 * poll-based here (ef_wait_timed ticks re-check EF_EVENTS), so no PPU
 * wait-slot registration is needed — and deliberately NOT written, so the
 * SPU-side lifted Set code takes its "no PPU waiter" path and simply ORs
 * bits that our poll then observes. */
/* The flag struct is ONE 128-byte line the SPU task library updates with
 * GETLLAR/PUTLLC. A PPU read-modify-write of it must hold the lock-line lock
 * for its whole span, or a PUTLLC landing between the read and the write is
 * silently undone: an SPU wait registered there is lost, nothing ever signals
 * the task, and its PPU partner blocks for good (GH3: FMOD's mixer deadlocked
 * at boot in 1 run of ~4). Raw accesses inside: vm_write16 would take the
 * same (non-recursive) lock. The end notifies reserving SPUs, under the lock
 * as spu_coh_notify_write expects. */
extern void spu_lockline_lock(void);
extern void spu_lockline_unlock(void);
extern int  spu_coh_is_reserved(uint32_t);
extern void spu_coh_notify_write(uint32_t);
static inline u16 ef_r16(uint32_t a) { const uint8_t* p = vm_base + a; return (u16)((p[0] << 8) | p[1]); }
static inline void ef_w16(uint32_t a, u16 v) { uint8_t* p = vm_base + a; p[0] = (uint8_t)(v >> 8); p[1] = (uint8_t)v; }
static inline void ef_line_end(uint32_t ea)
{
    if (spu_coh_is_reserved(ea & ~127u)) spu_coh_notify_write(ea & ~127u);
    spu_lockline_unlock();
}

static void spurs_ef_set_locked(uint32_t ea, u16 bits)
{
    spu_lockline_lock();
    u16 events        = ef_r16(ea + EF_EVENTS);
    u16 used          = (u16)(ef_r16(ea + EF_SPU_USED_SLOTS) &
                              ~ef_r16(ea + EF_SPU_PENDING_RECV));
    u16 waitmode      = ef_r16(ea + EF_SPU_WAIT_MODE);
    u16 eventsToClear = 0;
    u16 pendingRecv   = 0;

    for (int s = 0; s < CELL_SPURS_EVENT_FLAG_MAX_WAIT_SLOTS; s++) {
        u16 bit = (u16)(0x8000u >> s);
        if (!(used & bit)) continue;
        u16 mask = ef_r16(ea + EF_SPU_WAIT_MASK_ARR + 2u * s);
        u16 rel  = (u16)((events | bits) & mask);
        int mode_and = (waitmode & bit) != 0;
        if ((mask & ~rel) == 0 || (!mode_and && rel != 0)) {
            eventsToClear |= rel;
            pendingRecv   |= bit;
            ef_w16(ea + EF_PENDING_RECV_EVT + 2u * s, rel);
        }
    }

    events = (u16)(events | bits);
    if (pendingRecv) {
        ef_w16(ea + EF_SPU_PENDING_RECV,
               (u16)(ef_r16(ea + EF_SPU_PENDING_RECV) | pendingRecv));
        if (vm_read8(ea + EF_CLEAR_MODE) == CELL_SPURS_EVENT_FLAG_CLEAR_AUTO)
            events = (u16)(events & ~eventsToClear);
    }
    ef_w16(ea + EF_EVENTS, events);
    ef_line_end(ea);   /* before signalling: signal_task takes its own locks */

    for (int s = 0; s < CELL_SPURS_EVENT_FLAG_MAX_WAIT_SLOTS; s++) {
        if (pendingRecv & (0x8000u >> s)) {
            uint32_t taskset_ea = (uint32_t)vm_read64(ea + EF_ADDR);
            u32 taskId = vm_read8(ea + EF_WAITING_TASK_ID + s);
            if (ps3_log_verbose())
                fprintf(stderr, "[cellSpurs] EventFlagSet 0x%08X satisfies SPU task "
                        "slot %d (taskset=0x%08X task=%u)\n", ea, s, taskset_ea, taskId);
            spu_taskset_signal_task(taskset_ea, taskId);
        }
    }

    /* SPURS_EF_WAKE_PARKED: a task that parked in WAIT_SIGNAL WITHOUT registering
     * a wait slot in this flag is unreachable by the loop above, which only
     * signals slots present in `used`. YDKJ CRI tasks do exactly that, so a
     * PPU->SPU Set (direction 2) delivered nothing and they slept forever. Same
     * failure class canersaka documented for SPURS queues in Yakuza Dead Souls:
     * the HLE woke only its host condvar, which lifted SPU tasks never wait on.
     * Signalling a task that was not waiting on THIS flag is safe -- signals are
     * latched in guest state and every wait loop re-checks its own predicate. */
    /* Record the bits this Set owes the task waiting on (flag - 0x80),
     * unconditionally. Gating this on EF_DIRECTION made delivery depend on
     * Initialize having already run, and the CRI flags are Set before that --
     * so the post silently never happened. An unclaimed post is harmless: it is
     * only ever consumed by a task that names this exact object in its wait. */
    if (ea > 0x80u) {
        uint32_t obj = ea - 0x80u;
        /* Write NOW *and* record for consume time. The two orderings are both
         * real and neither alone is enough: if the task has not read yet the
         * immediate write is what it sees, and if it is parked (or took the
         * latched-wake path and never parked at all) the WAIT_SIGNAL handler
         * replays it just before returning. Writing the same bits twice is
         * idempotent, so covering both costs nothing and removes the race that
         * made this fire in only 1 of 4 runs. */
        { static int s_od = -1;
          if (s_od < 0) s_od = getenv("SPURS_EF_OBJ_DELIVER") ? 1 : 0;
          if (s_od) {
              vm_write16(obj + EF_EVENTS, bits);
              for (uint32_t s = 0; s < 16; s++)
                  vm_write16(obj + EF_PENDING_RECV_EVT + 2u * s, bits);
          } }
        { extern void spu_ef_bits_post(uint32_t, uint16_t);
          spu_ef_bits_post(obj, bits); }
    }
    if (!pendingRecv) {
        static int s_wp = -1;
        if (s_wp < 0) s_wp = getenv("SPURS_EF_WAKE_PARKED") ? 1 : 0;

        if (s_wp && vm_read8(ea + EF_DIRECTION) == 2) {
            uint32_t taskset_ea = (uint32_t)vm_read64(ea + EF_ADDR);
            /* The task waiting for THIS flag parked on object (flag - 0x80):
             * YDKJ CRI obj 0x006B4500/4780/4A00 <-> flag 0x006B4580/4800/4A80.
             * Fall back to any parked task of the taskset if none matches. */
            extern int spu_taskset_signal_parked_obj(uint32_t, uint32_t);

            int woke = spu_taskset_signal_parked_obj(taskset_ea, ea - 0x80u);
            if (!woke) woke = spu_taskset_signal_parked_obj(taskset_ea, 0);
            if (!woke) {   /* nobody parked yet -- do not lose the wakeup */
                extern void spu_taskset_latch_wake(uint32_t);
                spu_taskset_latch_wake(taskset_ea);
            }
            static int _n = 0;
            if (woke && _n++ < 12)
                fprintf(stderr, "[cellSpurs] EventFlagSet 0x%08X woke %d parked "
                                "task(s) on taskset 0x%08X (no wait slot)\n",
                        ea, woke, taskset_ea);
        }
    }
}

/* SPU-side entry (Layer 2): a task's flag Set arriving via the taskset
 * syscall or a runtime bridge. Same protocol, takes the lock itself. */
void spurs_ef_set_from_spu(uint32_t flag_ea, uint16_t bits)
{
    if (ftl_line() && (flag_ea & ~127u) == g_ftl_line) ps3_ftl(5, flag_ea, bits);   /* v115 */
    EventFlagSync* sync = ef_sync_get(flag_ea);
    if (!sync) return;
    ef_lock(sync);
    spurs_ef_set_locked(flag_ea, (u16)bits);
    ef_broadcast(sync);
    ef_unlock(sync);
}

/* =========================================================================
 * SPURS core
 *
 * The CellSpurs instance (0x2000 bytes of GAME-owned guest memory) is real
 * shared state: the game's engine pokes it with INLINED atomics (readyCount
 * stores, signal bits) and the policy modules DMA it from the SPU side. So
 * it must hold the REAL big-endian kernel layout — never a host struct.
 * Verified offsets (RPCS3 cellSpurs.h contract):
 *   +0x00 wklReadyCount1[16] (u8/wid)   +0x80 wklState1[16] (u8: 2=runnable)
 *   +0x10 wklIdleSpuCount[16]           +0x90 wklStatus1[16]
 *   +0x20 wklCurrentContention[16]      +0xA0 wklEvent1[16]
 *   +0x40 wklMinContention[16]          +0xB0 wklEnabled (be u32, bit 31-wid)
 *   +0x50 wklMaxContention[16]          +0xBD sysSrvMsgUpdateWorkload (u8)
 *   +0x60 wklFlag (be u64)              +0xB00 wklInfo1[16] (32B each:
 *   +0x70 wklSignal1 (be u16)                  addr u64, arg u64, size u32,
 *   +0x76 nSpus (u8)                           uniqueId u8, prio[8] @+0x18)
 * Our own bookkeeping lives in a host side-table keyed by instance EA.
 * A host "kernel" thread per instance polls readyCount/signal and runs the
 * workload's policy module (spurs_policy.c) — the virtual SPU.
 * =====================================================================*/
enum {
    SPURS_WKL_READY1   = 0x00,
    SPURS_WKL_IDLE2    = 0x10,
    SPURS_WKL_CURCONT  = 0x20,
    SPURS_WKL_MINCONT  = 0x40,
    SPURS_WKL_MAXCONT  = 0x50,
    SPURS_WKL_FLAG     = 0x60,
    SPURS_WKL_SIGNAL1  = 0x70,
    SPURS_NSPUS        = 0x76,
    SPURS_WKL_STATE1   = 0x80,
    SPURS_WKL_STATUS1  = 0x90,   /* v102: u8/wid, bit per SPU that has the workload */
    SPURS_WKL_EVENT1   = 0xA0,   /* v102: u8/wid, bit 0 = shutdown complete */
    SPURS_WKL_ENABLED  = 0xB0,
    SPURS_SYSSRV_MSG   = 0xBD,
    SPURS_WKL_INFO1    = 0xB00,
    SPURS_WKL_INFO_SZ  = 0x20,
    /* CELL_SPURS_SIZE = 4096 (SDK cell/spurs/types.h). The 8192-byte variant
     * is CellSpurs2 (cellSpursInitialize*2* NIDs) which LBP does not use —
     * clearing 0x2000 here overran the game's 4KB heap block and corrupted
     * the allocator (abort in the job pump's first object destruction). */
    SPURS_INST_SIZE    = 0x1000,
};

#define MAX_SPURS_INST 4
static struct SpursInst {
    u32           ea;          /* 0 = free */
    u32           nspus;
    char          prefix[16];
    volatile long kernel_live; /* poll thread started */
    int           solo;        /* v29: SPURS_WKL_SOLO threads in use */
} s_inst[MAX_SPURS_INST];

static struct SpursInst* spurs_inst_find(u32 ea)
{
    for (int i = 0; i < MAX_SPURS_INST; i++)
        if (s_inst[i].ea == ea) return &s_inst[i];
    return NULL;
}

struct SpursKArg { struct SpursInst* si; u32 part; u32 nparts; };

/* v29 SPURS_WKL_SOLO (default "0"; empty or "-" = off): comma-separated wids that each
 * get a DEDICATED kernel thread (threads 0..nsolo-1) on multi-SPU instances; the
 * other wids are spread over the remaining threads by wid % n. [pm-time] (with the
 * v29 window fix) showed w0 = CTaskMgr (data 0x404E9300, kicked via ReadyCountStore 5)
 * is the heaviest workload by ~2x and shared kernel thread 0 with w5/w10 -- real
 * work queued behind it. Only the partition changes: each workload still runs on
 * exactly one thread, lanes in order. SPURS_WKL_MAP="wid:thread,..." overrides any
 * single wid explicitly (thread index taken modulo the thread count). */
static int  s_solo[16];
static u32  s_nsolo = 0;
static int  s_map[16];
/* NB16 NB_WKL_MAP_B="wid:thread,..." (default unset): a second explicit map that replaces SPURS_WKL_MAP for the listed
 * wids while PS3_ABX bit 0x2000 is set -> in-run A/B of a partition change (the per-wid run flag keeps a wid on one
 * thread at a time across the switch, see v97f). */
static int  s_map_b[16];
extern volatile int g_abx;
static void spurs_wkl_map_init(void)
{
    static volatile long s_done = 0;
    if (s_done) return;
    for (int i = 0; i < 16; i++) s_map[i] = -1;   /* s_map_b: 0 = unset, else thread + 1 (NB16) */
    {   const char* mb = getenv("NB_WKL_MAP_B");
        if (mb) { const char* q = mb;
            while (*q) {
                while (*q == ',' || *q == ' ') q++;
                if (!*q) break;
                int w = atoi(q);
                while (*q && *q != ':' && *q != ',') q++;
                if (*q == ':') { q++; int t = atoi(q); if (w >= 0 && w < 16 && t >= 0) s_map_b[w] = t + 1; }
                while (*q && *q != ',') q++;
            }
            fprintf(stderr, "[NB] NB_WKL_MAP_B=%s (used while ABX 0x2000)\n", mb);
        } }
    const char* e = getenv("SPURS_WKL_SOLO");
    if (!e) e = "0";
    s_nsolo = 0;
    if (*e && *e != '-') {
        const char* q = e;
        while (*q && s_nsolo < 4) {
            while (*q == ',' || *q == ' ') q++;
            if (!*q) break;
            int w = atoi(q);
            if (w >= 0 && w < 16) s_solo[s_nsolo++] = w;
            while (*q && *q != ',') q++;
        }
    }
    const char* m = getenv("SPURS_WKL_MAP");
    if (m) {
        const char* q = m;
        while (*q) {
            while (*q == ',' || *q == ' ') q++;
            if (!*q) break;
            int w = atoi(q);
            while (*q && *q != ':' && *q != ',') q++;
            if (*q == ':') { q++; int t = atoi(q); if (w >= 0 && w < 16 && t >= 0) s_map[w] = t; }
            while (*q && *q != ',') q++;
        }
    }
    s_done = 1;
}
/* Kernel thread (0..nparts-1) that owns `wid` on an instance with `nparts` threads.
 * `solo` = this instance uses the solo threads (nparts includes them). */
/* v97f: PS3_AB phase 1 = "v97f features off" (also used by the lane-chain switch). On builds
 * without the PS3_AB thread (Linux), V97_AB_TOGGLE_SEC=<n> flips it every n seconds for testing. */
extern volatile int g_ab_mask;
static int v97_ab_off(void)
{
#ifndef _WIN32
    static int s_tog = -1;
    if (s_tog < 0) { const char* e = getenv("V97_AB_TOGGLE_SEC"); s_tog = e ? atoi(e) : 0; if (s_tog < 0) s_tog = 0; }
    if (s_tog) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (int)((ts.tv_sec / s_tog) & 1); }
#endif
    return g_ab_mask == 1;
}
/* v97f SPURS_SOLO_AB=<wid,...> (default "8,9"; "-" = none): SPURS_WKL_SOLO wids listed here keep
 * their dedicated kernel thread only OUTSIDE PS3_AB phase 1; in phase 1 they fall back to the
 * shared partition (wid % shared threads). play96 (SOLO 0,4 + 7 shared): CTaskMgr w8 shared t3 with
 * w1 (m_joblistMesh, 34% of t3) and CTaskMgr w9 shared t4 with w2 (BatchJobList, 1.1 ms runs, max
 * 5.5 ms, half in the SPU interpreter), so their passes queued behind the other workload's pass while
 * the PPU waited on them. play97 uses SPURS_WKL_SOLO=0,4,8,9 (phase 1 = the play96 pairing). A wid
 * still runs on exactly one thread at a time: every dispatch holds the wid's run flag (below). */
static u32 s_solo_ab_mask = 0;
static u32 spurs_solo_ab_init(void)
{
    static int done = 0;
    if (!done) {
        const char* e = getenv("SPURS_SOLO_AB");
        if (!e) e = "8,9";
        u32 m = 0;
        if (*e && *e != '-') { const char* q = e;
            while (*q) { while (*q == ',' || *q == ' ') q++; if (!*q) break;
                int w = atoi(q); if (w >= 0 && w < 16) m |= 1u << w;
                while (*q && *q != ',') q++; } }
        s_solo_ab_mask = m; done = 1;
    }
    return s_solo_ab_mask;
}
/* v97f: per-wid run flag -- a partition change (phase switch) must never let two kernel threads run
 * the same workload's lanes at once. Uncontended: one atomic exchange per dispatch. */
static volatile long s_wkl_run_flag[16];
static void spurs_wkl_run_lock(u32 wid)
{
    if (wid >= 16) return;
    for (unsigned k = 0; __atomic_exchange_n(&s_wkl_run_flag[wid], 1, __ATOMIC_ACQUIRE); k++) {
        if (k < 64) YieldProcessor(); else SwitchToThread();
    }
}
static void spurs_wkl_run_unlock(u32 wid)
{
    if (wid < 16) __atomic_store_n(&s_wkl_run_flag[wid], 0, __ATOMIC_RELEASE);
}
/* v102 (2026-10-08) workload lifecycle: ShutdownWorkload / WaitForWorkloadShutdown / RemoveWorkload
 * with the real libspurs state machine (RPCS3 cellSpurs.cpp + cellSpursSpu.cpp, SPURS kernel
 * system service):
 *   AddWorkload        state PREPARING(1) -> RUNNABLE(2); readyCount/status/event/signal of the wid cleared.
 *   ShutdownWorkload   RUNNABLE -> SHUTTING_DOWN(3) when an SPU still has the workload (wklStatus != 0),
 *                      else straight to REMOVABLE(4) + wklEvent |= 1. Already 3/4 -> CELL_OK.
 *   kernel             never dispatches a non-RUNNABLE workload; when the last SPU that ran a
 *                      SHUTTING_DOWN workload leaves it -> REMOVABLE + wklEvent |= 1 (+ completion event).
 *   WaitForWorkload..  blocks until the workload is REMOVABLE.
 *   RemoveWorkload     SHUTTING_DOWN -> CELL_SPURS_POLICY_MODULE_ERROR_BUSY (the game loops on 0x8041080A),
 *                      REMOVABLE -> NON_EXISTENT(0), wklEnabled bit cleared, else ERROR_STAT.
 * Until v101 Shutdown/Wait were no-ops and Remove only cleared the host in_use flag: a policy-module
 * quantum (all lanes + v97f lane chains) that was in flight kept running after the game had been
 * told the workload was gone and had freed its CTaskMgr arena, and the guest state stayed RUNNABLE.
 * Here "an SPU still has the workload" = the host run flag of the wid (s_wkl_run_flag, held by the
 * kernel thread for the whole quantum incl. its parallel lanes).
 * SPURS_WKL_LIFECYCLE=0 restores the v101 no-op behaviour. SPURS_WKL_WAIT_MS (default 5000): the
 * longest WaitForWorkloadShutdown waits for an in-flight quantum before it gives up (logged).
 * SPURS_WKL_STRICT=1: RemoveWorkload of a still-RUNNABLE workload returns ERROR_STAT like real
 * libspurs (default: implicit Shutdown + Wait, logged -- older titles may skip Shutdown).
 * SPURS_WKL_ADD_FRESH (default 1): AddWorkload of a reused wid starts from a clean host state
 * (persistent policy-module contexts of all lanes reset even when the new data EA equals the old one,
 * PM image re-resolved if the module changed, readyCount/signal/status/event bytes zeroed). */
enum { WKL_ST_NONE = 0, WKL_ST_PREPARING = 1, WKL_ST_RUNNABLE = 2, WKL_ST_SHUTTING_DOWN = 3, WKL_ST_REMOVABLE = 4 };
#define SPURS_PM_ERR_INVAL 0x80410802u
#define SPURS_PM_ERR_SRCH  0x80410805u
#define SPURS_PM_ERR_BUSY  0x8041080Au
#define SPURS_PM_ERR_STAT  0x8041080Fu
#define SPURS_PM_ERR_NULL  0x80410811u
static long long tw_now_us(void);
extern volatile long long g_spurs_tw_open_us;
static int spurs_wkl_lifecycle_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("SPURS_WKL_LIFECYCLE"); int v = (e && *e == '0') ? 0 : 1;
        fprintf(stderr, "[WKL] v102 SPURS_WKL_LIFECYCLE=%d (%s)\n", v,
                v ? "real Shutdown/Wait/Remove state machine, in-flight quanta finish first" : "v101: Shutdown/Wait no-ops");
        on = v;
    }
    return on;
}
static int spurs_wkl_add_fresh_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("SPURS_WKL_ADD_FRESH"); on = (e && *e == '0') ? 0 : 1; }
    return on;
}
/* Per-wid lifecycle telemetry (host only). */
static volatile long long s_wkl_add_us[16], s_wkl_sd_us[16], s_wkl_last_busy_us[16], s_wkl_last_run_us[16];
static volatile long s_wkl_busy_runs[16], s_wkl_runs[16], s_wkl_runs_after_sd[16], s_wkl_gen[16];
static volatile long s_wkl_log_n = 0;
static volatile unsigned char s_wkl_pm_stale[16];   /* AddWorkload: PM changed -> spurs_resolve_pm re-resolves */
static int spurs_wkl_log_ok(void) { return __atomic_add_fetch(&s_wkl_log_n, 1, __ATOMIC_RELAXED) <= 400; }
static double stl_t(void);
static double wkl_t(void) { return stl_t(); }   /* same clock as the [STALL] t= lines */
static u8 wkl_state(u32 ea, u32 wid) { return __atomic_load_n(vm_base + ea + SPURS_WKL_STATE1 + wid, __ATOMIC_SEQ_CST); }
/* BE u32 wklEnabled @0xB0: bit (31-wid) lives in byte wid/8, mask 0x80 >> (wid&7). */
static void wkl_enabled_bit(u32 ea, u32 wid, int set)
{
    u8* b = vm_base + ea + SPURS_WKL_ENABLED + (wid >> 3);
    const u8 m = (u8)(0x80u >> (wid & 7));
    if (set) __atomic_fetch_or(b, m, __ATOMIC_SEQ_CST); else __atomic_fetch_and(b, (u8)~m, __ATOMIC_SEQ_CST);
}
/* BE u16 wklSignal1 @0x70: bit (15-wid) in byte 0x70 + wid/8. */
static void wkl_clear_signal(u32 ea, u32 wid)
{
    __atomic_fetch_and(vm_base + ea + SPURS_WKL_SIGNAL1 + (wid >> 3), (u8)~(0x80u >> (wid & 7)), __ATOMIC_SEQ_CST);
}
static void wkl_dump_line(const char* tag, u32 wid, u32 data_ea)
{
    if (data_ea < 0x10000u || data_ea >= 0xF0000000u) return;
    const u8* p = vm_base + (data_ea & ~15u);
    for (u32 o = 0; o < 0xA0; o += 32) {
        fprintf(stderr, "[WKL]   %s w%u data+%02X:", tag, wid, o);
        for (u32 k = 0; k < 32; k += 4) fprintf(stderr, " %02X%02X%02X%02X", p[o+k], p[o+k+1], p[o+k+2], p[o+k+3]);
        fprintf(stderr, "\n");
    }
}
/* SHUTTING_DOWN -> REMOVABLE once no kernel thread holds the wid. holder=1: the caller holds the run
 * flag (kernel, after its quantum). Returns 1 when the workload is REMOVABLE. */
static int spurs_wkl_try_removable(u32 ea, u32 wid, int holder, const char* who)
{
    if (wid >= 16) return 0;
    u8 st = wkl_state(ea, wid);
    if (st != WKL_ST_SHUTTING_DOWN) return st == WKL_ST_REMOVABLE;
    if (!holder && __atomic_load_n(&s_wkl_run_flag[wid], __ATOMIC_SEQ_CST)) return 0;
    u8 exp = WKL_ST_SHUTTING_DOWN;
    if (__atomic_compare_exchange_n(vm_base + ea + SPURS_WKL_STATE1 + wid, &exp, (u8)WKL_ST_REMOVABLE, 0,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
        *(vm_base + ea + SPURS_WKL_STATUS1 + wid) = 0;
        __atomic_fetch_or(vm_base + ea + SPURS_WKL_EVENT1 + wid, (u8)1, __ATOMIC_SEQ_CST);   /* shutdown complete */
        if (spurs_wkl_log_ok())
            fprintf(stderr, "[WKL] t=%.3f wid %u SHUTTING_DOWN -> REMOVABLE by %s, %.1f ms after Shutdown, %ld quanta ran after Shutdown\n",
                    wkl_t(), wid, who, s_wkl_sd_us[wid] ? (tw_now_us() - s_wkl_sd_us[wid]) / 1000.0 : 0.0, s_wkl_runs_after_sd[wid]);
        return 1;
    }
    return wkl_state(ea, wid) == WKL_ST_REMOVABLE;
}
/* Kernel, right after taking the wid's run flag: may this quantum run? (state re-read under the flag,
 * so a Shutdown that saw the flag clear can never be followed by a run.) */
static int spurs_wkl_may_run(u32 ea, u32 wid, uint64_t arg)
{
    if (!spurs_wkl_lifecycle_on()) return 1;
    if (wkl_state(ea, wid) == WKL_ST_RUNNABLE && s_workloads[wid].in_use && s_workloads[wid].spurs_ea == ea &&
        vm_read64(ea + SPURS_WKL_INFO1 + wid * SPURS_WKL_INFO_SZ + 8) == arg) return 1;   /* arg: not a removed+re-added wid */
    spurs_wkl_try_removable(ea, wid, 1, "kernel (skipped quantum)");
    return 0;
}
/* Kernel, after a quantum (still holding the run flag). */
static void spurs_wkl_after_run(u32 ea, u32 wid, unsigned polls)
{
    if (wid >= 16) return;
    long long now = tw_now_us();
    s_wkl_runs[wid]++; s_wkl_last_run_us[wid] = now;
    if (polls) { s_wkl_busy_runs[wid]++; s_wkl_last_busy_us[wid] = now; }
    if (!spurs_wkl_lifecycle_on()) return;
    if (s_wkl_sd_us[wid]) s_wkl_runs_after_sd[wid]++;
    if (wkl_state(ea, wid) == WKL_ST_SHUTTING_DOWN) spurs_wkl_try_removable(ea, wid, 1, "kernel (quantum ended)");
}
/* v102 [WKL-QUIET]: called with the [pm-time] window counters BEFORE they are reset. A wid that
 * was busy in >= 3 windows in a row and found no work in a whole window, or an existing RUNNABLE
 * wid that was not dispatched at all, is named here (characters freezing = a CTaskMgr list that
 * stops getting or finding work). */
static void spurs_wkl_quiet_report(int sl, u32 ea, const unsigned* runs, const unsigned* busy, double win)
{
    static unsigned char s_streak[4][16], s_quiet[4][16], s_nodisp[4][16];
    if (sl < 0 || sl >= 4) return;
    for (u32 w = 0; w < 16; w++) {
        const int live = s_workloads[w].in_use && s_workloads[w].spurs_ea == ea;
        if (!live) { s_streak[sl][w] = s_quiet[sl][w] = s_nodisp[sl][w] = 0; continue; }
        const u8 st = wkl_state(ea, w);
        const int en = (vm_read32(ea + SPURS_WKL_ENABLED) >> (31 - w)) & 1;
        const u32 data = (u32)(s_workloads[w].data >> 32) ? (u32)(s_workloads[w].data >> 32) : (u32)s_workloads[w].data;
        if (!runs[w]) {
            if (st == WKL_ST_RUNNABLE && en && !s_nodisp[sl][w]) {
                s_nodisp[sl][w] = 1;
                fprintf(stderr, "[WKL-QUIET] t=%.1f inst %d wid %u (data 0x%08X) is RUNNABLE+enabled but was NOT DISPATCHED in the last %.1f s (run flag %ld)\n",
                        wkl_t(), sl, w, data, win, (long)s_wkl_run_flag[w]);
            }
            continue;
        }
        if (s_nodisp[sl][w]) { s_nodisp[sl][w] = 0; fprintf(stderr, "[WKL-QUIET] t=%.1f inst %d wid %u dispatched again\n", wkl_t(), sl, w); }
        if (busy[w]) {
            if (s_quiet[sl][w])
                fprintf(stderr, "[WKL-QUIET] t=%.1f inst %d wid %u (data 0x%08X) busy again (%u busy runs in %.1f s)\n", wkl_t(), sl, w, data, busy[w], win);
            s_quiet[sl][w] = 0;
            if (s_streak[sl][w] < 255) s_streak[sl][w]++;
        } else {
            if (s_streak[sl][w] >= 3 && !s_quiet[sl][w]) {
                s_quiet[sl][w] = 1;
                fprintf(stderr, "[WKL-QUIET] t=%.1f inst %d wid %u (data 0x%08X) was busy in %u windows in a row, found NO work in the last %.1f s (%u runs); state %u, last busy run %.1f s ago, PPU wait open %s\n",
                        wkl_t(), sl, w, data, s_streak[sl][w], win, runs[w], st,
                        s_wkl_last_busy_us[w] ? (tw_now_us() - s_wkl_last_busy_us[w]) / 1e6 : -1.0, g_spurs_tw_open_us ? "yes" : "no");
                wkl_dump_line("quiet", w, data);
            }
            s_streak[sl][w] = 0;
        }
    }
}
/* v102: real workload lifecycle (see spurs_wkl_lifecycle_on). spurs_ea = CellSpurs guest EA. */
static s32 spurs_wkl_shutdown_impl(u32 ea, u32 wid, const char* why)
{
    if (!ea) return (s32)SPURS_PM_ERR_NULL;
    if (wid >= CELL_SPURS_MAX_WORKLOAD) return (s32)SPURS_PM_ERR_INVAL;
    if (!s_workloads[wid].in_use || s_workloads[wid].spurs_ea != ea) return (s32)SPURS_PM_ERR_STAT;
    u8 exp = WKL_ST_RUNNABLE;
    if (!__atomic_compare_exchange_n(vm_base + ea + SPURS_WKL_STATE1 + wid, &exp, (u8)WKL_ST_SHUTTING_DOWN, 0,
                                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
        if (exp == WKL_ST_SHUTTING_DOWN || exp == WKL_ST_REMOVABLE) return CELL_OK;
        return (s32)SPURS_PM_ERR_STAT;
    }
    const long long now = tw_now_us();
    s_wkl_sd_us[wid] = now; s_wkl_runs_after_sd[wid] = 0;
    const long inflight = __atomic_load_n(&s_wkl_run_flag[wid], __ATOMIC_SEQ_CST);
    const u64 d = s_workloads[wid].data;
    const u32 dea = (u32)(d >> 32) ? (u32)(d >> 32) : (u32)d;
    if (spurs_wkl_log_ok()) {
        fprintf(stderr, "[WKL] t=%.3f Shutdown%s wid %u data 0x%08X: RUNNABLE -> SHUTTING_DOWN, quantum in flight %s, "
                        "last busy run %.1f ms ago, last run %.1f ms ago, lifetime %.1f s, %ld runs / %ld busy, PPU wait open %s\n",
                wkl_t(), why, wid, dea, inflight ? "YES" : "no",
                s_wkl_last_busy_us[wid] ? (now - s_wkl_last_busy_us[wid]) / 1000.0 : -1.0,
                s_wkl_last_run_us[wid] ? (now - s_wkl_last_run_us[wid]) / 1000.0 : -1.0,
                s_wkl_add_us[wid] ? (now - s_wkl_add_us[wid]) / 1e6 : -1.0,
                s_wkl_runs[wid], s_wkl_busy_runs[wid], g_spurs_tw_open_us ? "yes" : "no");
        wkl_dump_line("shutdown", wid, dea);
    }
    *(vm_base + ea + SPURS_SYSSRV_MSG) = 0xFF;            /* sysSrvMsgUpdateWorkload, as libspurs does */
    spurs_wkl_try_removable(ea, wid, 0, "Shutdown (no quantum in flight)");
    return CELL_OK;
}
static s32 spurs_wkl_wait_impl(u32 ea, u32 wid, const char* why)
{
    if (!ea) return (s32)SPURS_PM_ERR_NULL;
    if (wid >= CELL_SPURS_MAX_WORKLOAD) return (s32)SPURS_PM_ERR_INVAL;
    if (!s_workloads[wid].in_use || !((vm_read32(ea + SPURS_WKL_ENABLED) >> (31 - wid)) & 1)) return (s32)SPURS_PM_ERR_SRCH;
    u8 st = wkl_state(ea, wid);
    if (st == WKL_ST_RUNNABLE) {
        /* Real libspurs blocks until somebody shuts it down; nobody here would. */
        if (spurs_wkl_log_ok()) fprintf(stderr, "[WKL] t=%.3f Wait%s wid %u: still RUNNABLE (no Shutdown) -> returns at once\n", wkl_t(), why, wid);
        return CELL_OK;
    }
    static int s_ms = -1;
    if (s_ms < 0) { const char* e = getenv("SPURS_WKL_WAIT_MS"); s_ms = e ? atoi(e) : 5000; if (s_ms < 1) s_ms = 1; }
    const long long t0 = tw_now_us();
    unsigned k = 0;
    while (wkl_state(ea, wid) == WKL_ST_SHUTTING_DOWN) {
        if (spurs_wkl_try_removable(ea, wid, 0, "Wait")) break;
        if (tw_now_us() - t0 > (long long)s_ms * 1000) {
            fprintf(stderr, "[WKL] t=%.3f Wait%s wid %u: quantum still in flight after %d ms -> forcing REMOVABLE (SPURS_WKL_WAIT_MS)\n",
                    wkl_t(), why, wid, s_ms);
            u8 exp = WKL_ST_SHUTTING_DOWN;
            __atomic_compare_exchange_n(vm_base + ea + SPURS_WKL_STATE1 + wid, &exp, (u8)WKL_ST_REMOVABLE, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
            __atomic_fetch_or(vm_base + ea + SPURS_WKL_EVENT1 + wid, (u8)1, __ATOMIC_SEQ_CST);
            break;
        }
        if (++k < 256) YieldProcessor(); else if (k < 1024) SwitchToThread(); else Sleep(1);
    }
    const long long dt = tw_now_us() - t0;
    if (spurs_wkl_log_ok())
        fprintf(stderr, "[WKL] t=%.3f Wait%s wid %u: REMOVABLE after %.3f ms (%s)\n", wkl_t(), why, wid, dt / 1000.0,
                k ? "waited for the in-flight quantum" : "nothing in flight");
    return CELL_OK;
}
static u32 spurs_wkl_part(u32 wid, u32 nparts, int solo)
{
    if (wid < 16 && (g_abx & 0x2000) && s_map_b[wid] > 0) return (u32)(s_map_b[wid] - 1) % nparts;   /* NB16 */
    if (wid < 16 && s_map[wid] >= 0) return (u32)s_map[wid] % nparts;
    if (solo && s_nsolo && nparts > s_nsolo) {
        for (u32 i = 0; i < s_nsolo; i++) if ((u32)s_solo[i] == wid) {
            if (((spurs_solo_ab_init() >> wid) & 1) && v97_ab_off()) break;   /* v97f: shared in phase 1 */
            return i;
        }
        return s_nsolo + wid % (nparts - s_nsolo);
    }
    return wid % nparts;
}
static DWORD WINAPI spurs_kernel_thread(LPVOID p);
static void spurs_freeze_dump(const char* why);   /* v103B: F8 dump of every workload + lane */

static s32 spurs_initialize_common(u32 spurs_ea, u32 nspus, const char* prefix)
{
    { extern void (*volatile g_ps3_fz_spurs_dump)(const char*); g_ps3_fz_spurs_dump = spurs_freeze_dump; }   /* sys_ppu_thread.c */
    struct SpursInst* si = spurs_inst_find(spurs_ea);
    if (!si) {
        for (int i = 0; i < MAX_SPURS_INST; i++)
            if (!s_inst[i].ea) { si = &s_inst[i]; break; }
    }
    if (!si) return CELL_SPURS_CORE_ERROR_NOMEM;

    si->ea    = spurs_ea;
    si->nspus = (nspus > 0 && nspus <= CELL_SPURS_MAX_SPU) ? nspus : 1;
    memset(si->prefix, 0, sizeof(si->prefix));
    if (prefix) memcpy(si->prefix, prefix, 15);

    /* Real BE instance: zero it, then the few live fields. (The global
     * workload table is NOT wiped here — the title may init several SPURS
     * instances before adding workloads to any of them.) */
    memset(vm_base + spurs_ea, 0, SPURS_INST_SIZE);
    *(vm_base + spurs_ea + SPURS_NSPUS) = (u8)si->nspus;
    vm_write64(spurs_ea + SPURS_WKL_FLAG, 0xFFFFFFFFFFFFFFFFull); /* no receiver */

    if (!si->kernel_live) {
        si->kernel_live = 1;
        /* SPURS_WKL_THREADS (2026-10-04, default = nSpus, max 6): several
         * kernel threads per instance, each owning a FIXED subset of workloads
         * (wid % n == part). A workload only ever runs on its own thread and
         * still visits its lanes in order every pass (SEQ_LANES), so per-workload
         * behaviour is unchanged; different workloads run concurrently, as on
         * real SPUs. One thread ran all of inFamous' 9 workloads x 5 lanes in
         * turn and the main thread waited on it every frame. =1 -> old model. */
        /* v30: default 7 for a multi-SPU instance (was nSpus = 5), cap 12. Each
         * workload still runs on exactly ONE host thread with its lanes in order;
         * more threads only means fewer unrelated workloads queue behind each
         * other (v29: t2 = w1+w6+w11 and t5 = w4+w9 ran at ~90%). Host SPUs are
         * slower than real ones, so more host threads than SPUs is fine (28-thread CPU). */
        u32 nparts = si->nspus > 1 ? 7u : 1u;
        { const char* e = getenv("SPURS_WKL_THREADS"); if (e) nparts = (u32)atoi(e); }
        if (nparts < 1) nparts = 1;
        if (nparts > 12) nparts = 12;
        /* v29: + one dedicated thread per SPURS_WKL_SOLO wid (multi-SPU instances). */
        spurs_wkl_map_init();
        if (si->nspus > 1 && nparts > 1 && s_nsolo) { si->solo = 1; nparts += s_nsolo; }
        for (u32 t = 0; t < nparts; t++) {
            struct SpursKArg* ka = (struct SpursKArg*)malloc(sizeof *ka);
            ka->si = si; ka->part = t; ka->nparts = nparts;
            CreateThread(NULL, 1u << 20, spurs_kernel_thread, ka, 0, NULL);
        }
        fprintf(stderr, "[spurs-kern] \"%s\" %u kernel thread(s) (SPURS_WKL_THREADS, %u solo)\n",
                si->prefix, nparts, si->solo ? s_nsolo : 0u);
    }
    printf("[cellSpurs] Initialize \"%s\" ea=0x%08X nSpus=%u (real BE instance + kernel poll)\n",
           si->prefix, spurs_ea, si->nspus);
    return CELL_OK;
}

s32 cellSpursInitialize(CellSpurs* spurs, s32 nSpus, s32 spuPriority,
                        s32 ppuPriority, u8 exitIfNoWork)
{
    (void)spuPriority; (void)ppuPriority; (void)exitIfNoWork;
    if (!spurs)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    return spurs_initialize_common((u32)(uintptr_t)spurs, (u32)nSpus, NULL);
}

s32 cellSpursInitializeWithAttribute(CellSpurs* spurs,
                                     const CellSpursAttribute* attr)
{
    if (!spurs || !attr)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    u32 spurs_ea = (u32)(uintptr_t)spurs;
    attr = GUEST_PTR(attr, const CellSpursAttribute*);
    return spurs_initialize_common(spurs_ea, attr->nSpus, (const char*)attr->prefix);
}

static void spurs_cancel_attached_queues(u32 spurs_ea);  /* defined with the queue table */

s32 cellSpursFinalize(CellSpurs* spurs)
{
    if (!spurs)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;

    /* After this, no job completion can ever reach the attached queues, so a
     * worker parked on one with an infinite timeout would sleep forever -- and
     * the shutdown that called us then blocks in sys_ppu_thread_join waiting
     * for that worker to notice it should quit. Tokyo Jungle wedges exactly
     * there: "terminate audio thread (6)" then join(6), with tid 6 in
     * event_queue_receive(q=3) and the title never rendering again. Releasing
     * the waiters lets the receive fail, which is the path its loop already
     * handles -- an error return leaves the loop and the thread exits.
     *
     * Done BEFORE the instance lookup: that lookup fails here, and a failed
     * handle is no reason to strand a thread. */
    spurs_cancel_attached_queues((u32)(uintptr_t)spurs);

    struct SpursInst* si = spurs_inst_find((u32)(uintptr_t)spurs);
    if (!si)
        return CELL_SPURS_CORE_ERROR_STAT;

    printf("[cellSpurs] Finalize(ea=0x%08X)\n", si->ea);
    memset(s_workloads, 0, sizeof(s_workloads));
    si->ea = 0;   /* kernel thread sees a dead instance and idles */
    return CELL_OK;
}

s32 cellSpursAttributeInitialize(CellSpursAttribute* attr, s32 nSpus,
                                 s32 spuPriority, s32 ppuPriority,
                                 u8 exitIfNoWork)
{
    (void)ppuPriority; (void)exitIfNoWork;

    if (!attr)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;

    /* `attr` is a GUEST address (generic HLE adapter passes r3 raw). */
    attr = GUEST_PTR(attr, CellSpursAttribute*);
    memset(attr, 0, sizeof(CellSpursAttribute));
    attr->nSpus = (nSpus > 0 && nSpus <= CELL_SPURS_MAX_SPU)
                  ? (u32)nSpus : 1;

    for (int i = 0; i < CELL_SPURS_MAX_SPU; i++)
        attr->spuPriority[i] = spuPriority;

    printf("[cellSpurs] AttributeInitialize(nSpus=%d)\n", nSpus);
    return CELL_OK;
}

/* The SDK's cellSpursAttributeInitialize() macro imports this internal name
 * (NID 0x95180230). Forward to the implementation above. */
/* Real ABI (RPCS3 cellSpurs.cpp): _cellSpursAttributeInitialize(attr,
 * u32 revision, u32 sdkVersion, u32 nSpus, s32 spuPriority, s32 ppuPriority,
 * b8 exitIfNoWork). This used to take `revision` (2) as nSpus, so EVERY SPURS
 * instance got 2 SPUs: inFamous asks for 5 (instance 0x7CF100) and 1
 * (0x7D0380), as an RPCS3 guest-memory dump shows (+0x76). The game sizes its
 * WWS batches per SPU from cellSpursGetNumSpuThread, so its BatchJobList jobs
 * were built for a different SPU count than the instance reported.
 * PS3_SPURS_ATTR_LEGACY=1 restores the old (revision-as-nSpus) behaviour. */
s32 _cellSpursAttributeInitialize(CellSpursAttribute* attr, u32 revision,
                                  u32 sdkVersion, s32 nSpus, s32 spuPriority,
                                  s32 ppuPriority, u8 exitIfNoWork)
{
    static int s_legacy = -1;
    if (s_legacy < 0) s_legacy = getenv("PS3_SPURS_ATTR_LEGACY") ? 1 : 0;
    if (s_legacy)
        return cellSpursAttributeInitialize(attr, (s32)revision, (s32)sdkVersion,
                                            nSpus, (u8)spuPriority);
    printf("[cellSpurs] _AttributeInitialize(rev=%u sdk=0x%X nSpus=%d spuPrio=%d)\n",
           revision, sdkVersion, nSpus, spuPriority);
    return cellSpursAttributeInitialize(attr, nSpus, spuPriority,
                                        ppuPriority, exitIfNoWork);
}

s32 cellSpursAttributeSetNamePrefix(CellSpursAttribute* attr,
                                    const char* prefix, u32 size)
{
    if (!attr)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursAttribute*);
    const char* prefix_h = GUEST_PTR(prefix, const char*);

    if (prefix_h && size > 0) {
        u32 copyLen = size < sizeof(attr->prefix) ? size : sizeof(attr->prefix) - 1;
        memcpy(attr->prefix, prefix_h, copyLen);
        attr->prefix[copyLen] = '\0';
        attr->prefixSize = copyLen;
    }

    return CELL_OK;
}

s32 cellSpursAttributeSetSpuThreadGroupType(CellSpursAttribute* attr,
                                            s32 type)
{
    (void)type;
    if (!attr) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    return CELL_OK;
}

s32 cellSpursAttributeEnableSpuPrintfIfAvailable(CellSpursAttribute* attr)
{
    if (!attr) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    return CELL_OK;
}

s32 cellSpursGetNumSpuThread(const CellSpurs* spurs, u32* nThreads)
{
    if (!spurs || !nThreads)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    struct SpursInst* si = spurs_inst_find((u32)(uintptr_t)spurs);
    u32* nThreads_h = GUEST_PTR(nThreads, u32*);

    if (!si)
        return CELL_SPURS_CORE_ERROR_STAT;

    /* out-param is guest BE */
    vm_write32((u32)(uintptr_t)nThreads, si->nspus);
    (void)nThreads_h;
    return CELL_OK;
}

s32 cellSpursSetMaxContention(CellSpurs* spurs, CellSpursWorkloadId wid,
                              u32 maxContention)
{
    (void)maxContention;

    if (!spurs) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (wid >= CELL_SPURS_MAX_WORKLOAD) return CELL_SPURS_CORE_ERROR_INVAL;
    if (!s_workloads[wid].in_use) return CELL_SPURS_CORE_ERROR_SRCH;

    s_workloads[wid].maxContention = maxContention;
    return CELL_OK;
}

s32 cellSpursSetPriorities(CellSpurs* spurs, CellSpursWorkloadId wid,
                           const u8* priorities)
{
    if (!spurs || !priorities) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    const u8* priorities_h = GUEST_PTR(priorities, const u8*);
    if (wid >= CELL_SPURS_MAX_WORKLOAD) return CELL_SPURS_CORE_ERROR_INVAL;
    if (!s_workloads[wid].in_use) return CELL_SPURS_CORE_ERROR_SRCH;

    memcpy(s_workloads[wid].priority, priorities_h, CELL_SPURS_MAX_SPU);
    return CELL_OK;
}

/* The lv2 event queues the app attached to SPURS, and the port we handed back.
 * A title may attach more than one (Tokyo Jungle attaches two), so keep them
 * all -- storing a single id let the second attach hide the first. */
#define MAX_SPURS_QUEUES 8
#define SPURS_EVENT_PORT 0u
extern int sys_event_queue_push_by_id(uint32_t queue_id, uint64_t source,
                                      uint64_t data1, uint64_t data2, uint64_t data3);
extern void sys_event_queue_cancel_by_id(uint32_t queue_id);
extern void sys_event_queue_uncancel_by_id(uint32_t queue_id);

static u32 s_spurs_event_queue[MAX_SPURS_QUEUES];
/* WHICH SPURS instance attached each queue. A title with more than one instance
 * -- Tokyo Jungle boots one for its data install and keeps a second for audio --
 * finalises them separately, and cancelling every queue on any Finalize kills
 * the surviving instance's queues too. That reads to the title as "the SPU
 * stopped answering": its sound engine's receives fail with ECANCELED and it
 * tears itself down. Cancel only the queues the instance being finalised owns. */
static u32 s_spurs_event_queue_owner[MAX_SPURS_QUEUES];
static int s_spurs_event_queue_n = 0;

/* Release anyone blocked on the completion queues SPURS attached; see
 * cellSpursFinalize. */
static void spurs_cancel_attached_queues(u32 spurs_ea)
{
    int keep = 0;
    for (int i = 0; i < s_spurs_event_queue_n; i++) {
        /* owner 0 means "attached before we tracked owners" -- cancel those on
         * any finalize, which is the old behaviour and the safe default. */
        u32 owner = s_spurs_event_queue_owner[i];
        if (spurs_ea == 0 || owner == 0 || owner == spurs_ea) {
            sys_event_queue_cancel_by_id(s_spurs_event_queue[i]);
        } else {
            s_spurs_event_queue[keep]       = s_spurs_event_queue[i];
            s_spurs_event_queue_owner[keep] = owner;
            keep++;
        }
    }
    s_spurs_event_queue_n = keep;
}

/* SPU user-event port -> lv2 queue, for SPU code running under SPURS.
 *
 * A SPURS task signals the PPU with send_event(port, ...) exactly as a raw SPU
 * thread does, but there is no lv2 SPU thread group to hold the port binding,
 * so the lv2 delivery path had nowhere to look and dropped the event -- and
 * with it the ack the SPU then blocks on in SPU_RdInMbox. Havok's SPU task
 * stalls on the first physics step that way (port 2), and the PPU step waits
 * for it forever. */
static u32 s_spurs_port_queue[64];

uint32_t spurs_port_queue(uint32_t port)
{
    return port < 64 ? s_spurs_port_queue[port] : 0;
}

s32 cellSpursAttachLv2EventQueue(CellSpurs* spurs, u32 queue, u8* port,
                                 s32 isDynamic)
{
    /* KEEP the queue id. SPURS signals the application through the queue it
     * attaches here, and this discarded it -- so nothing SPURS ever did could
     * wake a thread sitting in sys_event_queue_receive on it. Tokyo Jungle
     * kicks its job chain and then blocks on exactly that receive. */
    if (queue && s_spurs_event_queue_n < MAX_SPURS_QUEUES) {
        /* A previous cellSpursFinalize may have cancelled this queue. Attaching
         * gives it a producer again, so lift that -- otherwise a title that
         * tears its audio down and brings it back finds every receive failing. */
        sys_event_queue_uncancel_by_id(queue);
        int dup = 0;
        for (int i = 0; i < s_spurs_event_queue_n; i++)
            if (s_spurs_event_queue[i] == queue) dup = 1;
        if (!dup) { s_spurs_event_queue_owner[s_spurs_event_queue_n] = (u32)(uintptr_t)spurs;
                    s_spurs_event_queue[s_spurs_event_queue_n++] = queue; }
    }

    if (!spurs || !port) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    u8* port_h = GUEST_PTR(port, u8*);

    /* Static: the caller names the port in *port. Dynamic: pick a free one,
     * 0x10..0x3F first (the low ports are the ones titles ask for by number),
     * and hand it back. This used to write 0 either way, so every attach
     * landed on port 0 and the SPU's real port number routed nowhere. */
    u32 p = *port_h;
    if (isDynamic) {
        p = 64;
        for (u32 i = 0x10; i < 64 && p == 64; i++) if (!s_spurs_port_queue[i]) p = i;
        for (u32 i = 0; i < 0x10 && p == 64; i++) if (!s_spurs_port_queue[i]) p = i;
        if (p == 64) return CELL_SPURS_CORE_ERROR_BUSY;
        *port_h = (u8)p;
    } else if (p >= 64) {
        return CELL_SPURS_CORE_ERROR_INVAL;
    } else if (s_spurs_port_queue[p]) {
        return CELL_SPURS_CORE_ERROR_BUSY;
    }
    s_spurs_port_queue[p] = queue;
    printf("[cellSpurs] AttachLv2EventQueue(queue=%u) port=%u%s\n", queue, p,
           isDynamic ? " (dynamic)" : "");
    return CELL_OK;
}

s32 cellSpursDetachLv2EventQueue(CellSpurs* spurs, u8 port)
{
    if (!spurs) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (port < 64) s_spurs_port_queue[port] = 0;
    printf("[cellSpurs] DetachLv2EventQueue(port=%u)\n", port);
    return CELL_OK;
}

/* =========================================================================
 * Taskset
 * =====================================================================*/

s32 cellSpursCreateTaskset(CellSpurs* spurs, CellSpursTaskset* taskset,
                           u64 args, const u8* priority, u32 maxContention)
{
    (void)args; (void)priority; (void)maxContention;

    /* Capture the GUEST EAs (raw register values) BEFORE host translation -- the real
     * BE taskset builder writes to guest memory at these EAs. */
    uint32_t taskset_ea = (uint32_t)(uintptr_t)taskset;
    uint32_t spurs_ea   = (uint32_t)(uintptr_t)spurs;

    /* Args arrive as guest effective addresses (ps3_hle_call passes raw guest
     * register values); translate to host before dereferencing. */
    taskset = GUEST_PTR(taskset, CellSpursTaskset*);

    if (!spurs || !taskset) {
        fprintf(stderr, "[cellSpurs] CreateTaskset REJECT null (spurs=0x%08X taskset=0x%08X)\n",
                spurs_ea, taskset_ea);
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    }

    if (!spurs_inst_find(spurs_ea)) {
        fprintf(stderr, "[cellSpurs] CreateTaskset REJECT unregistered spurs=0x%08X (taskset=0x%08X)\n",
                spurs_ea, taskset_ea);
        return CELL_SPURS_CORE_ERROR_STAT;
    }

    memset(taskset, 0, sizeof(CellSpursTaskset));
    taskset->initialized = 1;
    taskset->spurs = spurs;

    /* Write the REAL big-endian CellSpursTaskset layout (fork Option-B) so the lifted
     * SPU leaf + spurs_pm_build_context read valid data (the native writes above are
     * little-endian = garbage to the SPU). Overwrites 0x00-0x80 with BE fields. */
    spurs_taskset_init(taskset_ea, spurs_ea, args, /*wid*/0,
                       (uint32_t)sizeof(CellSpursTaskset), /*evf1*/0, /*evf2*/0);
    g_ydkj_real_taskset_ea = taskset_ea;

    g_ydkj_real_spurs_ea = spurs_ea;   /* capture for the taskset-policy handoff (LS[0x1C0]) */
    printf("[cellSpurs] CreateTaskset() ea=0x%08X spurs=0x%08X (real BE layout)\n", taskset_ea, spurs_ea);
    return CELL_OK;
}

s32 cellSpursCreateTasksetWithAttribute(CellSpurs* spurs,
                                        CellSpursTaskset* taskset,
                                        const CellSpursTasksetAttribute* attr)
{
    (void)attr;
    return cellSpursCreateTaskset(spurs, taskset, 0, NULL, 0);
}

s32 cellSpursDestroyTaskset(CellSpursTaskset* taskset)
{
    taskset = GUEST_PTR(taskset, CellSpursTaskset*);
    if (!taskset)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    printf("[cellSpurs] DestroyTaskset()\n");
    taskset->initialized = 0;
    return CELL_OK;
}

s32 cellSpursShutdownTaskset(CellSpursTaskset* taskset)
{
    taskset = GUEST_PTR(taskset, CellSpursTaskset*);
    if (!taskset)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    printf("[cellSpurs] ShutdownTaskset()\n");
    taskset->shutdownRequested = 1;
    return CELL_OK;
}

s32 cellSpursJoinTaskset(CellSpursTaskset* taskset)
{
    taskset = GUEST_PTR(taskset, CellSpursTaskset*);
    if (!taskset)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    printf("[cellSpurs] JoinTaskset()\n");
    /* In a full implementation, wait for all tasks to complete */
    return CELL_OK;
}

s32 cellSpursTasksetAttributeInitialize(CellSpursTasksetAttribute* attr)
{
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursTasksetAttribute*);
    memset(attr, 0, sizeof(CellSpursTasksetAttribute));
    attr->revision = 1;
    return CELL_OK;
}

s32 cellSpursTasksetAttributeSetName(CellSpursTasksetAttribute* attr,
                                      const char* name)
{
    (void)name;
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    return CELL_OK;
}

/* =========================================================================
 * Task
 * =====================================================================*/

/* Real SDK ABI (verified against RPCS3 cellSpurs.cpp:370):
 *   cellSpursCreateTask(taskset, taskId, elf, context, sizeContext,
 *                       CellSpursTaskLsPattern* lsPattern,
 *                       CellSpursTaskArgument*  argument)
 * -- SEVEN args (r3..r9), not six. The old 6-arg form treated r8 as an opaque
 * `attr` and never read r9 (the argument), so the task's 16-byte work-descriptor
 * argument was dropped and TaskInfo.args stayed 0. LBP's audio task DMAs its work
 * from an EA computed out of r3 = that argument (spu_0003 task main @0x17e70:
 * `wrch $ch18, f(r3.word3)`), so a zero argument makes it GET from EA 0 and
 * stall. lsPattern/argument arrive as guest EAs (generic adapter forwards r3..r10). */
s32 cellSpursCreateTask(CellSpursTaskset* taskset, CellSpursTaskId* taskId,
                        void* elf, void* context, u32 sizeContext,
                        u32 lsPattern_ea, u32 argument_ea)
{
    (void)context; (void)sizeContext;

    /* Read the 16-byte CellSpursTaskArgument + optional LS pattern from guest mem. */
    uint32_t task_arg[4] = {0,0,0,0};
    uint32_t task_lsp[4] = {0,0,0,0};
    if (argument_ea)  for (int _i=0;_i<4;_i++) task_arg[_i] = vm_read32(argument_ea + _i*4);
    if (lsPattern_ea) for (int _i=0;_i<4;_i++) task_lsp[_i] = vm_read32(lsPattern_ea + _i*4);

    /* Capture guest EAs BEFORE host translation (the real BE taskset builder + the
     * SPU DMA use guest EAs). */
    uint32_t taskset_ea = (uint32_t)(uintptr_t)taskset;
    uint32_t elf_ea     = (uint32_t)(uintptr_t)elf;
    uint32_t context_ea = (uint32_t)(uintptr_t)context;

    /* taskId/taskset are guest EAs; translate before deref. elf/context stay
     * guest EAs (handled below — elf is translated for load, context kept EA). */
    taskset = GUEST_PTR(taskset, CellSpursTaskset*);
    CellSpursTaskId* taskId_h = GUEST_PTR(taskId, CellSpursTaskId*);

    if (!taskset) {
        fprintf(stderr, "[cellSpurs] CreateTask REJECT null taskset (elf=0x%08X)\n", elf_ea);
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    }

    if (!g_ydkj_real_taskset_ea) { /* real-BE init flag (native ->initialized clobbered by BE layout) */
        fprintf(stderr, "[cellSpurs] CreateTask REJECT no-init (taskset=0x%08X elf=0x%08X)\n",
                taskset_ea, elf_ea);
        return CELL_SPURS_TASK_ERROR_STAT;
    }

    /* Find a free task slot IN THIS TASKSET: the first task id whose bit in the
     * taskset's enabled bitset (+0x30, 128 bits, task 0 = MSB) is clear. Task
     * exit clears that bit (spu_taskset_task_exited), so ids recycle the way
     * real SPURS recycles them. The old scan of the global s_tasks[] never
     * freed a slot unless the task was joined; Havok creates a task per step
     * and never joins, so after 128 steps CreateTask failed and the physics
     * step waited forever for a task that was never started.
     * ponytail: s_tasks[] is shared by every taskset and now only records the
     * last task per slot id -- enough for JoinTask2's lookup; key it by
     * (taskset, id) if a title joins tasks in two tasksets at once. */
    u32 free_slot = CELL_SPURS_MAX_TASK;
    for (u32 t = 0; t < CELL_SPURS_MAX_TASK; t++)
        if (!(vm_read8(taskset_ea + 0x30 + t / 8) & (0x80u >> (t % 8)))) { free_slot = t; break; }
    if (free_slot < CELL_SPURS_MAX_TASK) {
        u32 i = free_slot;
        {
            s_tasks[i].in_use = 1;
            s_tasks[i].id = s_next_task_id++;
            s_tasks[i].active = 1;
            s_tasks[i].completed = 0;
            s_tasks[i].exitCode = 0;
            s_tasks[i].entryPoint = elf;

            /* The guest reads this out-param BIG-ENDIAN, and a native store
             * put it in host order: task 1 came back to the game as 0x01000000,
             * which it then handed to _cellSpursSendSignal. No task by that id
             * exists, the signal was dropped, and the SPU task sat in
             * WAIT_SIGNAL while the PPU waited on the event flag it would have
             * set -- a two-sided deadlock from one missing byte swap.
             *
             * Hand back the SLOT INDEX, not the global counter: `i` is what
             * spurs_taskset_add_task() sets as the taskset's bitset bit and
             * what spu_taskset_signal_task() looks up, so the id the guest
             * signals with has to be the same number. */
            if (taskId) vm_write32((u32)(uintptr_t)taskId, i);
            (void)taskId_h;
            taskset->taskCount++;

            /* Register the task in the REAL BE taskset: writes task_info[slot]
             * (args/elf/context/ls_pattern) + sets enabled+ready bits so the PM's
             * SELECT_TASK picks it. Slot index i = the SPURS taskId (bitset bit). */
            spurs_taskset_add_task(taskset_ea, i, (uint64_t)elf_ea,
                                   (uint64_t)context_ea, task_arg, task_lsp);
            /* Bridge to the image-22 dispatch so build_context uses this taskset+task. */
            g_ydkj_real_taskset_ea = taskset_ea;
            g_ydkj_real_taskid     = i;

            printf("[cellSpurs] CreateTask(id=%u, entry=%p, arg=%08X %08X %08X %08X"
                   " ctx=0x%08X lsp=%08X %08X %08X %08X)\n",
                   s_tasks[i].id, elf, task_arg[0], task_arg[1], task_arg[2], task_arg[3],
                   context_ea, task_lsp[0], task_lsp[1], task_lsp[2], task_lsp[3]);

            /* One-shot: dump the memory the task argument points at, to find the
             * pointer that reads back 0 (the task GETs from EA 0 -> some field of
             * its work descriptor is null in our run). Each of the 4 arg words that
             * looks like a valid guest EA gets 64 bytes dumped as BE u32s. */
            if (getenv("SPURS_TASKSET_TRACE")) {
                for (int a = 0; a < 4; a++) {
                    uint32_t p = task_arg[a];
                    if (p < 0x10000 || p >= 0x50000000u) continue;   /* not a plausible EA */
                    fprintf(stderr, "[argdump] arg[%d]=0x%08X ->", a, p);
                    for (int o = 0; o < 64; o += 4)
                        fprintf(stderr, " %08X", vm_read32(p + o));
                    fprintf(stderr, "\n");
                }
                /* Full heap object (arg[2], the audio engine object): the count the
                 * PPU validated is at +0xC8 (16-aligned, <=0x160 = stream count); the
                 * task's per-stream buffer pointers live at +0x124 (=arg[1]). Dump
                 * +0x00..+0x160 so we can see: 0 streams (task should yield) vs N
                 * streams with null data buffers (fill gap). */
                uint32_t ho = task_arg[2];
                if (ho >= 0x10000 && ho < 0x50000000u) {
                    fprintf(stderr, "[heapobj] 0x%08X count@+0xC8=0x%08X\n", ho, vm_read32(ho + 0xC8));
                    for (int o = 0; o < 0x160; o += 16)
                        fprintf(stderr, "  +%03X: %08X %08X %08X %08X\n", o,
                                vm_read32(ho+o), vm_read32(ho+o+4), vm_read32(ho+o+8), vm_read32(ho+o+12));
                }
                /* The descriptor block the SPU task actually DMAs + reads its buffer
                 * pointers from: v10[336..] at v10+1344 = arg[3]-64 (arg[3]=v10+1408).
                 * v10[344]=a1[141], v10[345]=a1[140] (the FMOD DSP buffers). If those
                 * words are 0 here, they are the null source (task GETs from EA 0). */
                uint32_t d = task_arg[3];
                if (d >= 0x10040 && d < 0x50000000u) {
                    uint32_t db = d - 64;   /* 0x0094F6C0 = v10+1344 */
                    fprintf(stderr, "[descblk] v10+1344=0x%08X (a1[141]@+0x20, a1[140]@+0x24):\n", db);
                    for (int o = 0; o < 0x40; o += 16)
                        fprintf(stderr, "  +%02X: %08X %08X %08X %08X\n", o,
                                vm_read32(db+o), vm_read32(db+o+4), vm_read32(db+o+8), vm_read32(db+o+12));
                }
                /* a1 (the FMOD object) = taskId_ea - 628 (sub_48420C passes the taskId
                 * out-param as (_DWORD)a1+628 for task 0). a1[140]/a1[141] (= a1+0x230/
                 * +0x234) are the null DSP-buffer fields. Log the EAs so the next run
                 * can YDKJ_WWATCH=<a1+0x230> to catch who should write it (or prove no
                 * one does). Only for task 0 (offset 628); task 1 uses +688. */
                uint32_t a1 = (uint32_t)(uintptr_t)taskId - 628u;
                if (a1 < 0x50000000u)
                    fprintf(stderr, "[a1obj] a1=0x%08X  a1+0x230(dsp0)=0x%08X val=0x%08X  "
                            "a1+0x234(dsp1)=0x%08X val=0x%08X\n", a1,
                            a1+0x230, vm_read32(a1+0x230), a1+0x234, vm_read32(a1+0x234));
                fflush(stderr);
            }

            /* Run the task's SPU program if a lifted build is registered for it.
             * The registry maps the task ELF (by content fingerprint) to its
             * pre-lifted native entry; dispatch loads the ELF into a local store
             * and runs it with the task arg in r3. INERT until the title
             * registers its lifted SPU set: an unregistered image MISSes and
             * returns 0, preserving the prior "track only" behaviour.
             *
             * NOTE: dispatch is synchronous (runs to completion inline). That
             * suits create+join task patterns; a workload/taskset whose SPU job
             * waits on concurrent PPU-side signals will want the async lv2
             * SPU-thread path instead — wired when a title exercises it. */
            if (elf) {
                /* elf/context are guest effective addresses; translate the image
                 * pointer to host memory for fingerprint+load, but keep context
                 * as the guest EA (the SPU job's DMA uses guest EAs / r3). */
                const uint8_t* host_elf = GUEST_PTR(elf, const uint8_t*);
                size_t sz = spu_elf_image_size(host_elf, 2u * 1024 * 1024);
                if (sz)
                    /* Async: SPURS tasks are persistent workers — running them
                     * inline would block this PPU thread forever (deadlock). */
                    spu_workload_dispatch_async(host_elf, (uint32_t)sz,
                                                (uint32_t)(uintptr_t)context);
            }
            return CELL_OK;
        }
    }

    fprintf(stderr, "[cellSpurs] CreateTask: taskset 0x%08X has no free task id (all 128 enabled)\n",
            taskset_ea);
    return CELL_SPURS_TASK_ERROR_NOMEM;
}

/* The SDK's versioned task-attribute initializer. ABI (8 GPR args):
 *   r3=attr r4=revision r5=sdkVersion r6=eaElf r7=eaContext r8=sizeContext
 *   r9=lsPattern r10=argument
 * Stash the task ELF EA + context so cellSpursCreateTaskWithAttribute can
 * dispatch the SPU job. */
s32 _cellSpursTaskAttributeInitialize(CellSpursTaskAttribute* attr, u32 revision,
                                      u32 sdkVersion, u64 eaElf, u64 eaContext,
                                      u32 sizeContext, const void* lsPattern,
                                      const void* argument)
{
    (void)sdkVersion;
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursTaskAttribute*);
    memset(attr, 0, sizeof(CellSpursTaskAttribute));
    attr->revision    = revision;
    attr->sizeContext = sizeContext;
    attr->eaContext   = eaContext;
    attr->eaElf       = eaElf;
    /* lsPattern/argument are guest EAs of 16-byte blocks; carry them so
     * CreateTaskWithAttribute writes them into the TaskInfo. The SPU task
     * library refuses blocking waits for a task whose argument is zero or
     * whose lsPattern doesn't cover its stack (0x8041090F). */
    attr->lsPattern_ea = (u32)(uintptr_t)lsPattern;
    attr->argument_ea  = (u32)(uintptr_t)argument;

    /* SPURS_TASKATTR_R8: some callers use a SIX-argument form, passing only
     * r3..r8 -- r9/r10 then hold caller leftovers, not arguments. YDKJ is one:
     * func_00331DA4 sets r3..r8 and nothing else, so our 8-parameter prototype
     * reads lsPattern=r9 (a size, 0x0003D400) and argument=r10 (0x2E, the ASCII
     * terminator its hex-string parser stopped on). r8 is the real 16-byte
     * CellSpursTaskArgument -- proven by its contents matching the quadword the
     * SPU task later receives in r3. A zero/garbage argument makes the SPU task
     * library refuse every blocking wait (0x8041090F), which is exactly how the
     * CRI tasks end up parked in WAIT_SIGNAL forever.
     * CONFIRMED by a second, independently lifted title: Jackbox Party Pack
     * calls the same NID with the same shape -- r3=attr, r4=1, r5=0x00330000,
     * r6=eaElf, r7=sp+0x180, r8=sp+0x170 (two adjacent 16-byte blocks), and
     * r9/r10 never written. Two unrelated callers agreeing settles it, so this
     * is now the default; SPURS_TASKATTR_LEGACY=1 restores the old 8-arg read. */
    { static int s_r8 = -1;
      if (s_r8 < 0) s_r8 = getenv("SPURS_TASKATTR_LEGACY") ? 0 : 1;
      if (s_r8) {
          attr->argument_ea  = (u32)sizeContext;   /* r8 */
          attr->lsPattern_ea = 0;                  /* r9 was a leftover */
          printf("[cellSpurs] TaskAttr R8-form: argument_ea=0x%08X (was 0x%08X)\n",
                 attr->argument_ea, (u32)(uintptr_t)argument);
      } }
    /* SPURS_TASKATTR_DESC: a THIRD caller shape, found in Saints Row 2. Its
     * taskset builder (func_009F56E0) does not pass eaContext/sizeContext/
     * lsPattern as separate arguments at all -- r7 points at a 3-word
     * DESCRIPTOR it fills immediately before the call, and r8 is the argument:
     *
     *   009F58D0  addi r7, r1, 128     ; r7 = sp+0x80  -> the descriptor
     *   009F58D4  addi r8, r1, 176     ; r8 = sp+0xB0  -> CellSpursTaskArgument
     *   009F58E8  stw  r9,  0x80(r1)   ;   [0] context EA
     *   009F58EC  stw  r11, 0x84(r1)   ;   [1] context size, straight out of
     *                                  ;       cellSpursTaskGetContextSaveAreaSize
     *   009F58F0  stw  r25, 0x88(r1)   ;   [2] CellSpursTaskLsPattern*
     *
     * Read the 8-argument way, sizeContext comes out as r8 -- a STACK ADDRESS,
     * not a size (the log shows szctx=267382952 = 0x0FEFF0A8) -- and lsPattern
     * comes out of r9, which this caller never sets, so it is 0. Those are
     * exactly the conditions the SPU task library refuses to run a blocking
     * task under, and the tasks then park in WAIT_SIGNAL forever doing 0 ms of
     * work apiece, which is the symptom this title shows. Off by default: the
     * R8-form above is right for YDKJ and Jackbox and they are unaffected. */
    { static int s_desc = -1;
      if (s_desc < 0) s_desc = getenv("SPURS_TASKATTR_DESC") ? 1 : 0;
      if (s_desc) {
          uint32_t d = (uint32_t)eaContext;          /* r7 */
          attr->eaContext    = vm_read32(d + 0);
          attr->sizeContext  = vm_read32(d + 4);
          attr->lsPattern_ea = vm_read32(d + 8);
          attr->argument_ea  = (u32)sizeContext;     /* r8 */
          printf("[cellSpurs] TaskAttr DESC-form: desc=0x%08X -> ctx=0x%08X size=%u lsp=0x%08X arg=0x%08X\n",
                 d, (u32)attr->eaContext, attr->sizeContext,
                 attr->lsPattern_ea, attr->argument_ea);
      } }
    printf("[cellSpurs] _TaskAttributeInitialize(eaElf=0x%08X ctx=0x%08X szctx=%u lsp=0x%08X arg=0x%08X)\n",
           (u32)eaElf, (u32)eaContext, sizeContext,
           attr->lsPattern_ea, attr->argument_ea);
    return CELL_OK;
}

/* Create a task from a pre-initialized attribute (carries ELF EA + context).
 * Forwards to cellSpursCreateTask, which translates taskset/taskId and runs the
 * SPU image through spu_workload_dispatch. */
s32 cellSpursCreateTaskWithAttribute(CellSpursTaskset* taskset,
                                     CellSpursTaskId* taskId,
                                     CellSpursTaskAttribute* attr)
{
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    CellSpursTaskAttribute* attr_h = GUEST_PTR(attr, CellSpursTaskAttribute*);
    /* Dump the raw attribute: our struct doesn't model lsPattern/argument, and
     * a wait-capable task NEEDS its context size + ls pattern carried through
     * (a no-context task may not block -- SPU task-lib waits then fail with
     * ERROR_STAT). Learn the real field offsets from the bytes. */
    { uint32_t aea = (uint32_t)(uintptr_t)attr;
      static int _n = 0; if (_n++ < 6) {
        fprintf(stderr, "[cellSpurs] CreateTaskWithAttr attr=0x%08X raw:", aea);
        for (int o = 0; o < 0x40; o += 4) fprintf(stderr, " %08X", vm_read32(aea + o));
        fprintf(stderr, "\n"); } }
    /* taskset/taskId forwarded raw (callee translates); elf/context are guest
     * EAs, as are lsPattern/argument (stored by _cellSpursTaskAttributeInitialize;
     * dropping them left the TaskInfo with a zero argument + zero lsPattern and
     * the SPU task library then refuses every blocking wait with 0x8041090F --
     * LBP's binkspu movie-IO task spun forever on that). */
    return cellSpursCreateTask(taskset, taskId,
                               (void*)(uintptr_t)(u32)attr_h->eaElf,
                               (void*)(uintptr_t)(u32)attr_h->eaContext,
                               attr_h->sizeContext,
                               attr_h->lsPattern_ea, attr_h->argument_ea);
}

/* The SDK's versioned taskset-attribute initializer. We forward taskset creation
 * through CreateTaskset (which ignores the attribute), so just zero the struct. */
s32 _cellSpursTasksetAttributeInitialize(CellSpursTasksetAttribute* attr,
                                         u32 revision, u32 sdkVersion, u64 argTaskset,
                                         u64 priority, u32 maxContention)
{
    (void)sdkVersion; (void)argTaskset; (void)priority; (void)maxContention;
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursTasksetAttribute*);
    memset(attr, 0, sizeof(CellSpursTasksetAttribute));
    attr->revision = revision ? revision : 1;
    printf("[cellSpurs] _TasksetAttributeInitialize(rev=%u)\n", revision);
    return CELL_OK;
}

s32 cellSpursJoinTask(CellSpursTaskset* taskset, CellSpursTaskId taskId,
                      s32* exitCode)
{
    (void)taskset;
    s32* exitCode_h = GUEST_PTR(exitCode, s32*);

    printf("[cellSpurs] JoinTask(id=%u)\n", taskId);

    /* Find the task and mark as completed */
    for (u32 i = 0; i < CELL_SPURS_MAX_TASK; i++) {
        if (s_tasks[i].in_use && s_tasks[i].id == taskId) {
            s_tasks[i].completed = 1;
            s_tasks[i].active = 0;
            if (exitCode_h)
                *exitCode_h = s_tasks[i].exitCode;
            s_tasks[i].in_use = 0;
            return CELL_OK;
        }
    }

    return CELL_SPURS_TASK_ERROR_SRCH;
}

s32 cellSpursSendSignal(CellSpursTaskset* taskset, CellSpursTaskId taskId)
{
    /* Capture the guest taskset EA before host translation -- the WAIT_SIGNAL
     * waiter (spu_taskset_wait_signal) keys on the guest EA + taskId. */
    uint32_t taskset_ea = (uint32_t)(uintptr_t)taskset;

    printf("[cellSpurs] SendSignal(taskset=0x%08X id=%u)\n", taskset_ea, taskId);

    /* Deliver the signal for real: set the task's bit in the guest taskset's
     * SIGNALLED bitset and wake its blocked host thread. (Was a documented
     * no-op from the pre-SPU-execution era -- a dropped signal deadlocked any
     * task parked in WAIT_SIGNAL waiting for it.) */
    if (taskset_ea) spu_taskset_signal_task(taskset_ea, taskId);
    return CELL_OK;
}

s32 cellSpursTaskAttributeInitialize(CellSpursTaskAttribute* attr)
{
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursTaskAttribute*);
    memset(attr, 0, sizeof(CellSpursTaskAttribute));
    attr->revision = 1;
    return CELL_OK;
}

/* =========================================================================
 * Workload
 * =====================================================================*/

s32 cellSpursAddWorkload(CellSpurs* spurs, CellSpursWorkloadId* wid,
                         const void* pm, u32 sizePm, u64 data,
                         const u8* priority, u32 minContention,
                         u32 maxContention)
{
    if (!spurs || !wid)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    /* spurs/wid/priority are guest EAs; pm stays a guest EA (it's the SPU
     * program address consumed later by the workload dispatch). */
    uint32_t spurs_ea = (uint32_t)(uintptr_t)spurs;
    struct SpursInst* si = spurs_inst_find(spurs_ea);
    const u8* priority_h = GUEST_PTR(priority, const u8*);

    if (!si)
        return CELL_SPURS_CORE_ERROR_STAT;

    for (u32 i = 0; i < CELL_SPURS_MAX_WORKLOAD; i++) {
        int _free = 0;
        if (!s_workloads[i].in_use && __atomic_compare_exchange_n(&s_workloads[i].in_use, &_free, 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
            /* v102: a reused wid starts from a clean host + guest state (SPURS_WKL_ADD_FRESH). The
             * guest state byte is still NON_EXISTENT (Remove) or stale, so publish it last. */
            const int reused = s_wkl_gen[i] > 0;
            if (spurs_wkl_lifecycle_on()) *(vm_base + spurs_ea + SPURS_WKL_STATE1 + i) = WKL_ST_PREPARING;
            if (reused && spurs_wkl_add_fresh_on()) {
                extern void spurs_pm_force_fresh(uint32_t);
                spurs_pm_force_fresh(i);   /* every lane reloads the PM into a zeroed LS, even for the same data EA */
                if (s_workloads[i].pm != pm) s_wkl_pm_stale[i] = 1;   /* re-resolve the PM image (spurs_resolve_pm) */
            }
            if (spurs_wkl_add_fresh_on()) {
                *(vm_base + spurs_ea + SPURS_WKL_READY1 + i) = 0;
                *(vm_base + spurs_ea + SPURS_WKL_IDLE2 + i) = 0;
                *(vm_base + spurs_ea + SPURS_WKL_CURCONT + i) = 0;
                *(vm_base + spurs_ea + SPURS_WKL_STATUS1 + i) = 0;
                *(vm_base + spurs_ea + SPURS_WKL_EVENT1 + i) = 0;
                wkl_clear_signal(spurs_ea, i);
            }
            s_wkl_gen[i]++;
            s_wkl_add_us[i] = tw_now_us(); s_wkl_sd_us[i] = 0; s_wkl_runs_after_sd[i] = 0;
            s_wkl_runs[i] = 0; s_wkl_busy_runs[i] = 0; s_wkl_last_busy_us[i] = 0; s_wkl_last_run_us[i] = 0;
            s_workloads[i].pm = pm;
            s_workloads[i].sizePm = sizePm;
            s_workloads[i].data = data;
            s_workloads[i].spurs_ea = spurs_ea;
            s_workloads[i].minContention = minContention;
            s_workloads[i].maxContention = maxContention;
            s_workloads[i].readyCount = 0;

            if (priority_h)
                memcpy(s_workloads[i].priority, priority_h, CELL_SPURS_MAX_SPU);
            else
                memset(s_workloads[i].priority, 0, CELL_SPURS_MAX_SPU);

            /* Publish the workload in the REAL BE instance so the game's
             * inlined kernel protocol (readyCount stores, signal bits, state
             * reads) and the policy module's own instance DMAs see it. */
            u32 info = spurs_ea + SPURS_WKL_INFO1 + i * SPURS_WKL_INFO_SZ;
            vm_write64(info + 0x00, (u64)(uintptr_t)pm);        /* addr */
            vm_write64(info + 0x08, data);                       /* arg  */
            vm_write32(info + 0x10, sizePm);                     /* size */
            vm_write32(info + 0x14, i << 24);                    /* uniqueId */
            for (int b = 0; b < 8; b++)
                *(vm_base + info + 0x18 + b) = priority_h ? priority_h[b] : 0;
            *(vm_base + spurs_ea + SPURS_WKL_MINCONT + i) = (u8)(minContention ? minContention : 1);
            *(vm_base + spurs_ea + SPURS_WKL_MAXCONT + i) = (u8)(maxContention ? maxContention : 1);
            wkl_enabled_bit(spurs_ea, i, 1);
            __atomic_store_n(vm_base + spurs_ea + SPURS_WKL_STATE1 + i, (u8)WKL_ST_RUNNABLE, __ATOMIC_SEQ_CST);   /* runnable (published last) */
            *(vm_base + spurs_ea + SPURS_SYSSRV_MSG) = 0xFF;
            if (s_wkl_gen[i] > 1 && spurs_wkl_log_ok())
                fprintf(stderr, "[WKL] t=%.3f Add wid %u (reuse #%ld) data 0x%016llX pm 0x%08X: %s\n", wkl_t(), i, s_wkl_gen[i] - 1,
                        (unsigned long long)data, (u32)(uintptr_t)pm,
                        spurs_wkl_add_fresh_on() ? "fresh PM contexts on every lane, readyCount/signal/status/event cleared" : "v101 (SPURS_WKL_ADD_FRESH=0)");

            /* wid out-param is guest BE */
            vm_write32((u32)(uintptr_t)wid, i);
            printf("[cellSpurs] AddWorkload(wid=%u, pm=%p, size=%u)\n",
                   i, pm, sizePm);
            return CELL_OK;
        }
    }

    return CELL_SPURS_CORE_ERROR_NOMEM;
}

/* Real (BE) CellSpursWorkloadAttribute offsets (libspurs layout; the game's
 * inlined SDK code writes the struct directly in guest memory, so it must be
 * read back big-endian at these offsets -- never through a native host struct
 * (see the BIG-ENDIAN WARNING in spurs_taskset.h). */
enum {
    WKATTR_REVISION   = 0x00,   /* be u32 */
    WKATTR_SDKVERSION = 0x04,   /* be u32 */
    WKATTR_PM         = 0x08,   /* be u32: policy-module image EA */
    WKATTR_SIZE       = 0x0C,   /* be u32: policy-module size */
    WKATTR_DATA       = 0x10,   /* be u64: workload data (jobchain/queue EA) */
    WKATTR_PRIORITY   = 0x18,   /* u8[8] */
    WKATTR_MIN_CONT   = 0x20,   /* be u32 */
    WKATTR_MAX_CONT   = 0x24,   /* be u32 */
    WKATTR_NAME_CLASS = 0x28,   /* be u32: char* EA */
    WKATTR_NAME_INST  = 0x2C,   /* be u32: char* EA */
    WKATTR_HOOK       = 0x30,   /* be u32 */
    WKATTR_HOOK_ARG   = 0x34,   /* be u32 */
};

static volatile u32 s_pmwatch_ea = 0, s_pmwatch_sz = 0;
static DWORD WINAPI pm_write_watch(LPVOID unused)
{
    (void)unused;
    u32 ea = s_pmwatch_ea, sz = s_pmwatch_sz;
    u32 last_exe = vm_read32(ea + 0xAF4), last_hot = vm_read32(ea + 0x7E4); /* 0x14f4, 0x11e4 */
    for (int t = 0; t < 6000; t++) {          /* ~60s at 10ms */
        u32 exe = vm_read32(ea + 0xAF4), hot = vm_read32(ea + 0x7E4);
        if ((exe != last_exe && exe != 0) || (hot != last_hot && hot != 0xFFFFFFFFu)) {
            fprintf(stderr, "[pm-watch] PM ASSEMBLED at ~%dms: exe@0x14f4 %08X->%08X hot@0x11e4 %08X->%08X\n",
                    t * 10, last_exe, exe, last_hot, hot);
            FILE* pf = fopen("lbp_spu/pm_wwsjob_complete.bin", "wb");
            if (pf) { fwrite(vm_base + ea, 1, sz, pf); fclose(pf);
                fprintf(stderr, "[pm-watch] wrote lbp_spu/pm_wwsjob_complete.bin (0x%X bytes)\n", sz); }
            return 0;
        }
        Sleep(10);
    }
    fprintf(stderr, "[pm-watch] PM NEVER assembled in 60s (exe@0x14f4=%08X hot@0x11e4=%08X) "
            "-> assembly is HLE'd/external\n", vm_read32(ea + 0xAF4), vm_read32(ea + 0x7E4));
    return 0;
}

s32 cellSpursAddWorkloadWithAttribute(CellSpurs* spurs,
                                       CellSpursWorkloadId* wid,
                                       const CellSpursWorkloadAttribute* attr)
{
    if (!attr) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    uint32_t attr_ea = (uint32_t)(uintptr_t)attr;

    /* Decode the REAL BE attribute from guest memory. */
    u32 pm_ea = vm_read32(attr_ea + WKATTR_PM);
    u32 pm_sz = vm_read32(attr_ea + WKATTR_SIZE);
    u64 data  = vm_read64(attr_ea + WKATTR_DATA);
    u32 minc  = vm_read32(attr_ea + WKATTR_MIN_CONT);
    u32 maxc  = vm_read32(attr_ea + WKATTR_MAX_CONT);
    u32 nmcls = vm_read32(attr_ea + WKATTR_NAME_CLASS);
    u32 nmins = vm_read32(attr_ea + WKATTR_NAME_INST);

    {   /* Layout ground truth: dump the raw attr words for the first few calls
         * (if the decode above prints nonsense, these bytes are the arbiter). */
        static int _n = 0;
        if (_n < 3) {
            printf("[cellSpurs] AddWorkloadWA attr=0x%08X raw:", attr_ea);
            for (int o = 0; o < 0x40; o += 4) {
                if ((o & 15) == 0) printf("\n    +%02X:", o);
                printf(" %08X", vm_read32(attr_ea + o));
            }
            printf("\n");
        } else if (_n == 3) {
            printf("[cellSpurs] AddWorkloadWA raw dumps suppressed from here\n");
        }
        printf("[cellSpurs] AddWorkloadWA: pm=0x%08X size=%u data=0x%016llX minC=%u maxC=%u name=%s/%s\n",
               pm_ea, pm_sz, (unsigned long long)data, minc, maxc,
               nmcls ? (const char*)(vm_base + nmcls) : "-",
               nmins ? (const char*)(vm_base + nmins) : "-");
        if (_n == 0 && pm_ea && pm_ea < 0x10000000u) {
            printf("[cellSpurs] PM@0x%08X first 96B:", pm_ea);
            for (int o = 0; o < 96; o += 4) {
                if ((o & 15) == 0) printf("\n    +%02X:", o);
                printf(" %08X", vm_read32(pm_ea + o));
            }
            printf("\n");
        }
        /* PM-COMPLETENESS PROBE (SPURS_PM_DUMP): the wwsjob job-manager PM is
         * ASSEMBLED at runtime -- the embedded ELF (LS 0xA00) has zero HOLES at
         * LS 0xAEE..0x11B0 and the executeStage lives at LS 0x14f4, both filled
         * by SPURS setup. Report whether OUR runtime's PM has that code or the
         * holes, and (once) write the whole image out so we can re-lift it. */
        if (getenv("SPURS_PM_DUMP") && pm_ea && pm_sz >= 0x2200 && pm_sz <= 0x4000) {
            u32 exe = vm_read32(pm_ea + 0xAF4);          /* LS 0x14f4 executeStage */
            int zeros = 0; for (u32 o = 0xEE; o < 0x7B0; o += 4)
                if (vm_read32(pm_ea + o) == 0) zeros += 4;
            fprintf(stderr, "[pm-dump] wid-PM ea=0x%08X sz=0x%X exe@+0xAF4=%08X (want 24F880ED) "
                    "holeZeros=%d/1730 -> %s\n", pm_ea, pm_sz, exe, zeros,
                    (exe == 0x24F880EDu ? "COMPLETE (we assemble it)"
                                        : zeros > 1000 ? "HOLEY (assembly skipped)" : "PARTIAL"));
            static int _dumped = 0;
            if (!_dumped) { _dumped = 1;
                FILE* pf = fopen("lbp_spu/pm_wwsjob_runtime.bin", "wb");
                if (pf) { fwrite(vm_base + pm_ea, 1, pm_sz, pf); fclose(pf);
                    fprintf(stderr, "[pm-dump] wrote lbp_spu/pm_wwsjob_runtime.bin (0x%X bytes)\n", pm_sz); }
                /* WRITE-WATCH: spawn a poller that reports if/when the PM's
                 * placeholder hot-loop (LS 0x11e4, currently 0xFFFFFFFF) and
                 * executeStage (LS 0x14f4, currently 0) get real code written by
                 * PPU code at runtime -> tells us whether our recompiled game
                 * assembles the PM (dump it) or the assembly is HLE'd away. */
                s_pmwatch_ea = pm_ea; s_pmwatch_sz = pm_sz;
                CreateThread(NULL, 1u << 18, pm_write_watch, NULL, 0, NULL);
            }
        }
        _n++;
    }

    /* Forward the BE-decoded values (priority as guest EA of the 8-byte table). */
    return cellSpursAddWorkload(spurs, wid, (const void*)(uintptr_t)pm_ea,
                               pm_sz, data,
                               (const u8*)(uintptr_t)(attr_ea + WKATTR_PRIORITY),
                               minc, maxc);
}

/* The SDK-versioned workload-attribute initializer (the import the game links;
 * NID differs from the non-underscore inline wrapper). Args arrive raw in
 * r3..r10; writes the REAL BE layout so AddWorkloadWithAttribute round-trips. */
s32 _cellSpursWorkloadAttributeInitialize(u64 attr_ea, u32 revision, u32 sdkVersion,
                                          u64 pm_ea, u32 size, u64 data,
                                          u64 prio_ea, u32 minContention)
{
    if (!attr_ea) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    vm_write32((u32)attr_ea + WKATTR_REVISION,   revision);
    vm_write32((u32)attr_ea + WKATTR_SDKVERSION, sdkVersion);
    vm_write32((u32)attr_ea + WKATTR_PM,         (u32)pm_ea);
    vm_write32((u32)attr_ea + WKATTR_SIZE,       size);
    vm_write64((u32)attr_ea + WKATTR_DATA,       data);
    for (int i = 0; i < 8; i++)
        *(vm_base + (u32)attr_ea + WKATTR_PRIORITY + i) =
            prio_ea ? *(vm_base + (u32)prio_ea + i) : 0;
    vm_write32((u32)attr_ea + WKATTR_MIN_CONT, minContention);
    vm_write32((u32)attr_ea + WKATTR_MAX_CONT, 1);   /* 9th arg is beyond the 8-GPR adapter */
    vm_write32((u32)attr_ea + WKATTR_NAME_CLASS, 0);
    vm_write32((u32)attr_ea + WKATTR_NAME_INST,  0);
    vm_write32((u32)attr_ea + WKATTR_HOOK,     0);
    vm_write32((u32)attr_ea + WKATTR_HOOK_ARG, 0);
    printf("[cellSpurs] _WorkloadAttributeInitialize(attr=0x%08X pm=0x%08X size=%u data=0x%llX minC=%u)\n",
           (u32)attr_ea, (u32)pm_ea, size, (unsigned long long)data, minContention);
    return CELL_OK;
}

s32 cellSpursWorkloadAttributeSetName(u64 attr_ea, u64 nameClass_ea, u64 nameInstance_ea)
{
    if (!attr_ea) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    vm_write32((u32)attr_ea + WKATTR_NAME_CLASS, (u32)nameClass_ea);
    vm_write32((u32)attr_ea + WKATTR_NAME_INST,  (u32)nameInstance_ea);
    printf("[cellSpurs] WorkloadAttributeSetName(attr=0x%08X, \"%s\", \"%s\")\n",
           (u32)attr_ea,
           nameClass_ea ? (const char*)(vm_base + (u32)nameClass_ea) : "-",
           nameInstance_ea ? (const char*)(vm_base + (u32)nameInstance_ea) : "-");
    return CELL_OK;
}

s32 cellSpursRemoveWorkload(CellSpurs* spurs, CellSpursWorkloadId wid)
{
    if (!spurs) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (wid >= CELL_SPURS_MAX_WORKLOAD) return CELL_SPURS_CORE_ERROR_INVAL;
    if (!s_workloads[wid].in_use) return CELL_SPURS_CORE_ERROR_SRCH;

    if (spurs_wkl_lifecycle_on()) {   /* v102: real libspurs RemoveWorkload */
        const u32 ea = (u32)(uintptr_t)spurs;
        if (s_workloads[wid].spurs_ea != ea) return (s32)SPURS_PM_ERR_SRCH;
        u8 st = wkl_state(ea, wid);
        if (st == WKL_ST_RUNNABLE) {
            static int s_strict = -1;
            if (s_strict < 0) { const char* e = getenv("SPURS_WKL_STRICT"); s_strict = (e && *e == '1') ? 1 : 0; }
            if (s_strict) return (s32)SPURS_PM_ERR_STAT;
            spurs_wkl_shutdown_impl(ea, wid, " (implicit, Remove without Shutdown)");
            spurs_wkl_wait_impl(ea, wid, " (implicit)");
            st = wkl_state(ea, wid);
        }
        if (st == WKL_ST_SHUTTING_DOWN && !spurs_wkl_try_removable(ea, wid, 0, "Remove")) {
            static long s_busy_n = 0;
            if (++s_busy_n <= 20 || (s_busy_n % 1000) == 0)
                fprintf(stderr, "[WKL] t=%.3f Remove wid %u: still SHUTTING_DOWN -> ERROR_BUSY (#%ld)\n", wkl_t(), wid, s_busy_n);
            return (s32)SPURS_PM_ERR_BUSY;
        }
        st = wkl_state(ea, wid);
        if (st != WKL_ST_REMOVABLE) return (s32)SPURS_PM_ERR_STAT;
        u8 exp = WKL_ST_REMOVABLE;
        if (!__atomic_compare_exchange_n(vm_base + ea + SPURS_WKL_STATE1 + wid, &exp, (u8)WKL_ST_NONE, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
            return (s32)SPURS_PM_ERR_STAT;
        wkl_enabled_bit(ea, wid, 0);
        *(vm_base + ea + SPURS_WKL_READY1 + wid) = 0;
        *(vm_base + ea + SPURS_WKL_CURCONT + wid) = 0;
        wkl_clear_signal(ea, wid);
        const long long now = tw_now_us();
        if (spurs_wkl_log_ok())
            fprintf(stderr, "[WKL] t=%.3f Remove wid %u data 0x%016llX: REMOVABLE -> NON_EXISTENT, enabled bit cleared; lifetime %.1f s, %ld runs / %ld busy, %ld quanta after Shutdown, last busy run %.1f ms ago\n",
                    wkl_t(), wid, (unsigned long long)s_workloads[wid].data, s_wkl_add_us[wid] ? (now - s_wkl_add_us[wid]) / 1e6 : -1.0,
                    s_wkl_runs[wid], s_wkl_busy_runs[wid], s_wkl_runs_after_sd[wid],
                    s_wkl_last_busy_us[wid] ? (now - s_wkl_last_busy_us[wid]) / 1000.0 : -1.0);
        s_wkl_sd_us[wid] = 0;
    }
    __atomic_store_n(&s_workloads[wid].in_use, 0, __ATOMIC_SEQ_CST);
    printf("[cellSpurs] RemoveWorkload(wid=%u)\n", wid);
    return CELL_OK;
}

s32 cellSpursWorkloadAttributeInitialize(CellSpursWorkloadAttribute* attr,
                                         u32 revision, u32 sdkVersion,
                                         const void* pm, u32 sizePm,
                                         u64 data, const u8* priority,
                                         u32 minContention,
                                         u32 maxContention)
{
    if (!attr) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    attr = GUEST_PTR(attr, CellSpursWorkloadAttribute*);
    const u8* priority_h = GUEST_PTR(priority, const u8*);

    memset(attr, 0, sizeof(CellSpursWorkloadAttribute));
    attr->revision = revision;
    attr->sdkVersion = sdkVersion;
    attr->pm = (u64)(uintptr_t)pm;   /* pm kept as guest EA */
    attr->sizePm = sizePm;
    attr->data = data;
    attr->minContention = minContention;
    attr->maxContention = maxContention;

    if (priority_h)
        memcpy(attr->priority, priority_h, CELL_SPURS_MAX_SPU);

    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * The SPURS "kernel": one host poll thread per instance (the virtual SPU).
 *
 * The game kicks work by storing a nonzero wklReadyCount1[wid] byte or a
 * wklSignal1 bit into the instance — mostly with INLINED atomics (LBP never
 * calls an API for it beyond cellSpursReadyCountStore). The kernel thread
 * polls those real BE fields, consumes one ready unit (decrement / clear the
 * signal bit, like the real kernel's dispatch), and runs the workload's
 * policy module to completion via spu_run_policy_module.
 * -----------------------------------------------------------------------*/
typedef struct {
    spu_lifted_entry_fn fn;
    int                 image_id;
    int                 resolved;   /* 0=not tried, 1=found, -1=missing */
} WklPm;
static WklPm s_wkl_pm[CELL_SPURS_MAX_WORKLOAD];

static WklPm* spurs_resolve_pm(u32 wid)
{
    WklPm* r = &s_wkl_pm[wid];
    if (s_wkl_pm_stale[wid]) { s_wkl_pm_stale[wid] = 0; r->resolved = 0; }   /* v102: wid re-added with another PM */
    if (r->resolved) return r->resolved > 0 ? r : NULL;
    SpursWorkload* w = &s_workloads[wid];
    uint64_t fp = spu_workload_fingerprint(vm_base + (uint32_t)(uintptr_t)w->pm,
                                           w->sizePm);
    r->fn = spu_workload_find_img(fp, &r->image_id);
    r->resolved = r->fn ? 1 : -1;
    if (r->fn)
        printf("[cellSpurs] wid=%u PM resolved (fp=0x%016llX image=%d)\n",
               wid, (unsigned long long)fp, r->image_id);
    else
        printf("[cellSpurs] wid=%u PM NOT LIFTED (fp=0x%016llX size=%u) -- workload will not run\n",
               wid, (unsigned long long)fp, w->sizePm);
    return r->fn ? r : NULL;
}

/* Thread creation, waiting and Sleep come from runtime/platform/win32_compat.h
 * off Windows (included at the top of this file), which is what lets the three
 * thread bodies below stand as written on every host. SetThreadStackGuarantee
 * has no POSIX counterpart -- it reserves stack for the stack-overflow
 * exception handler, and there is no such handler here -- so it stays behind
 * an _WIN32 guard where it is used.
 *
 * One virtual SPU running a workload's policy module. The WWS job manager
 * runs concurrently across N SPUs (RPCS3: jobmanagerCellSpursKernel0..N): each
 * claims jobs from the shared queue and advances its own lane of the sync
 * barrier. Running the SPUs SEQUENTIALLY deadlocks -- SPU 0 completes its job,
 * then busy-waits at the cross-SPU barrier for lane 1, which the sequential
 * loop can never advance (SPU 1 hasn't run). Concurrency is required, not an
 * optimization: SPU 0's barrier poll observes SPU 1's atomic lane update live. */
struct spurs_pm_worker_arg {
    spu_lifted_entry_fn fn; int image_id;
    const uint8_t* pm; uint32_t pm_size;
    uint64_t arg; uint32_t wid, ea, spu_num;
};
static DWORD WINAPI spurs_pm_worker(LPVOID p)
{
#ifdef _WIN32
    { ULONG g = 256 * 1024; SetThreadStackGuarantee(&g); }
#endif
    struct spurs_pm_worker_arg* a = (struct spurs_pm_worker_arg*)p;
    spu_run_policy_module(a->fn, a->image_id, a->pm, a->pm_size,
                          a->arg, a->wid, a->ea, a->spu_num);
    return 0;
}

/* v30 SPURS_PAR_LANES=<wid,...> (EXPERIMENTAL, default off): run the listed
 * workloads' virtual-SPU lanes 1..n-1 CONCURRENTLY with lane 0 on persistent
 * per-workload worker threads (one wake + one join per run, no CreateThread).
 * [pm-time] in the v29 run: w0 (main CTaskMgr) alone keeps its kernel thread at
 * 90-100% and the PPU waits on it every frame. On hardware the attribute says
 * maxContention 1 (one SPU at a time), so this goes BEYOND the hardware -- the
 * WWS job manager is built for several SPUs on one list (per-SPU lanes, atomic
 * claims), but an older parallel kernel (SPURS_PAR_SPUS) deadlocked m_joblistMesh.
 * Try `SPURS_PAR_LANES=0`; if the game hangs / glitches, unset it. */
struct LanePool {
    SRWLOCK lk; CONDITION_VARIABLE cv_go, cv_done;
    volatile long gen, done; u32 nworkers;
    struct spurs_pm_worker_arg job;
    unsigned polls[8];
    unsigned long long spin[8];          /* v30c: spin-polls per worker lane (this run) */
    /* v30c stats (reported with [pm-time]): per lane, runs that polled + host ms */
    volatile long long st_ms_q[8]; volatile long st_busy[8], st_runs;
    volatile long st_kicks, st_rot_busy;   /* v31: rotate runs that woke the helpers / rotate runs that found work */
    volatile unsigned long long kick_tsc;  /* v96a [LANEPROF]: rdtsc at the last gen++ (wake latency) */
    volatile long ch_epoch;                /* v97f lane chain: busy lane runs completed (any lane) */
    volatile long st_ch_re, st_ch_idle;    /* v97f stats: chained re-runs, of which re-runs that found nothing */
    CONDITION_VARIABLE cv_hot;             /* v132 ABX 0x02: idle lanes of a round wait here for a busy run to finish */
    volatile long hot_active;              /* v132: lanes currently inside spurs_lane_run_chain (hot mode) */
    volatile long hot_idle;                /* v132: of those, lanes waiting idle on cv_hot (under lk) */
    volatile long st_hot_wait, st_hot_rerun; /* v132 stats */
};
/* v132 factorial A/B bits (ppu_loader.cpp PS3_ABX_SEQ): 0x02 hot idle lanes, 0x04 wait-targeted boost. */
extern volatile int g_abx;
/* v132 ABX 0x04: open hooked PPU waits per waited workload ([16] = not attributed). */
volatile long g_tw_wid_n[17];
static inline int spurs_tw_boost_wid(u32 w) { return (g_abx & 0x04) && w < 16 && g_tw_wid_n[w] > 0; }   /* attributed waits only */
/* v96a [LANEPROF] lane-pool timing (spurs_policy.c prints it): [wid] = {wake ticks, wakes, join-wait ticks, joins} */
extern volatile long long g_lp_pool[16][4];
extern int spu_laneprof_on(void);
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#ifndef _WIN32
#include <unistd.h>   /* sysconf (v96a) */
#endif
extern SPU_THREAD_LOCAL unsigned long long g_spu_spin_polls;
static struct LanePool* s_lp_all[16];
/* v96a SPURS_LANE_SPIN_US (adaptive default; 0 = off; also off in PS3_AB phase 1 = "worker A off").
 * A parallel pass hands lanes 1..n-1 to the pool workers through a condition variable and the
 * kernel joins them through another, so EVERY pass paid two OS thread wake-ups (the workers'
 * start and the kernel's join) on top of the slowest lane -- and a CTaskMgr / WWS list whose
 * jobs depend on each other needs one pass per dependency step (play95: w1 ~1400 passes/s,
 * ~155 us of lane work each). Now a worker whose last lane run found work spin-waits (PAUSE)
 * up to this long for the next pass before it sleeps, and the kernel spin-waits up to this long
 * for the join. Scheduling only: what each lane runs, and in what order, is unchanged.
 * Busy-waiting helps high-core desktops but steals scarce scheduling time on
 * low-end CPUs: <=8 logical CPUs default to 0, 9-15 to 50 us, and >=16 to
 * the measured desktop value of 200 us. */
extern volatile int g_ab_mask;
static int spurs_lane_spin_us(void)
{
    static int us = -2;
    if (us == -2) {
        const char* e = getenv("SPURS_LANE_SPIN_US");
        int v = e ? atoi(e) : 200;
        if (!e) {
#ifdef _WIN32
            DWORD ncpu = GetActiveProcessorCount(0xFFFF);
            if (ncpu <= 8) v = 0;
            else if (ncpu < 16) v = 50;
#else
            long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
            if (ncpu <= 8) v = 0;
            else if (ncpu < 16) v = 50;
#endif
        }
        if (v < 0) v = 0; if (v > 5000) v = 5000;
        fprintf(stderr, "[spurs-kern] v96a SPURS_LANE_SPIN_US=%d (lane pool spin-waits for the next pass / the join; 0 off; off in PS3_AB phase 1)\n", v);
        us = v;
    }
    { static int s_ab = -1; if (s_ab < 0) { const char* e = getenv("SPURS_LANE_SPIN_AB"); s_ab = (e && *e == '1') ? 1 : 0; }   /* v97f: phase 1 now belongs to v97f */
      if (g_abx & 0x08) return us < 50 ? us : 50;   /* v132 ABX 0x08 "less spinning": lane spin 200 -> 50 us */
      return (s_ab && g_ab_mask == 1) ? 0 : us; }
}
/* HP4B (2026-11-10) SPURS_SPIN_MODE (default 0 = old SwitchToThread spin):
 *   pause|1 : the kernel threads' spin body (after a busy pass, and inside the SPURS_SPIN_US window) is a burst of
 *             SPURS_SPIN_PAUSES (default 32) PAUSE instructions instead of a SwitchToThread() syscall; every 32nd burst still
 *             calls SwitchToThread() so an over-subscribed CPU is given away (R5 #5). Same windows, same nap policy.
 *   ab      : pause mode while g_abx & 0x8000000 (PS3_AB/PS3_ABX_SEQ in-run A/B; bit claimed by HP4B).
 *   NL_AB=SPURSSPIN inverts it in the backend's NL_AB phase B (g_hp4b_ab_phase, written by the D3D12 backend).
 * g_hp4b_ab_phase / hp4b_nl_ab_lists are also used by cellAudio.c (AUDIO_HRWAIT). */
volatile int g_hp4b_ab_phase = 0;
int hp4b_nl_ab_lists(const char* name)
{
    const char* l = getenv("NL_AB");
    if (!l || !name) return 0;
    const size_t n = strlen(name);
    for (const char* q = l; (q = strstr(q, name)) != NULL; q += n)
        if ((q == l || q[-1] == ',') && (q[n] == 0 || q[n] == ',')) return 1;
    return 0;
}
static int s_hp4b_np = 32;
static int hp4b_spurs_pause_mode(void)
{
    static int v = -1, lc = -1;
    if (v < 0) {
        const char* e = getenv("SPURS_SPIN_MODE");
        const char* p = getenv("SPURS_SPIN_PAUSES");
        int np = p ? atoi(p) : 32; if (np < 1) np = 1; if (np > 4096) np = 4096; s_hp4b_np = np;
        lc = hp4b_nl_ab_lists("SPURSSPIN");
        v = !e ? 0 : !strcmp(e, "ab") ? 2 : (!strcmp(e, "pause") || *e == '1') ? 1 : 0;
        fprintf(stderr, "[HP4B] SPURS_SPIN_MODE=%s (%s), %d pauses/burst%s\n", e ? e : "(unset)",
                v == 2 ? "pause while ABX 0x8000000" : v ? "PAUSE bursts" : "old SwitchToThread", np,
                lc > 0 ? ", NL_AB=SPURSSPIN inverts it in phase B" : "");
    }
    if (v == 2) { extern volatile int g_abx; return (g_abx & 0x8000000) != 0; }
    return (lc > 0 && g_hp4b_ab_phase) ? !v : v;
}
static void hp4b_spurs_spin(void)
{
    static SPU_PM_TLS unsigned k = 0;
    if (!hp4b_spurs_pause_mode()) { SwitchToThread(); return; }
    for (int z = 0; z < s_hp4b_np; z++) _mm_pause();
    if ((++k & 31u) == 0) SwitchToThread();
}
/* Spin (PAUSE) until *p != v (ne=1) or *p >= v (ne=0), at most us microseconds. */
static void spurs_lane_spin(volatile long* p, long v, int ne, int us)
{
    static LONGLONG s_f = 0;
    if (!s_f) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); s_f = f.QuadPart; }
    LARGE_INTEGER q; QueryPerformanceCounter(&q);
    const LONGLONG dl = q.QuadPart + (LONGLONG)us * s_f / 1000000;
    for (unsigned k = 1; ; k++) {
        long c = *p;
        if (ne ? (c != v) : (c >= v)) return;
        _mm_pause();
        if ((k & 31) == 0) { QueryPerformanceCounter(&q); if (q.QuadPart >= dl) return; }
    }
}
static int spurs_stats_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("SPURS_PMTIME"); on = (e && *e == '0') ? 0 : 1; }
    return on;
}
/* ---- QC 2026-10-16 (w2 critical path): QC_TL=<sec>[,<sec>...] (default unset = off; costs one load per hook):
 * microsecond timeline of the SPURS kernels, lane pools and hooked PPU task waits in windows of QC_TL_MS (default 3000,
 * max 20000) ms starting at those [FREEZE]-clock seconds (max 8 windows). Event types (column 1 of the dump):
 *   K kernel workload run (any wid; lane = kernel part, a = polls, b = arg lo, c = maxcont)
 *   R lane-pool round (spurs_run_lanes_parallel; a = summed polls, b = arg lo, c = lanes)
 *   Q rotate run (spurs_run_lane_kickable; a = summed polls, b = 1 if the other lanes were kicked, c = rotate lane)
 *   C lane chain (spurs_lane_run_chain; lane = virtual SPU, a = module runs, b = summed polls)
 *   M one policy-module run inside a chain (a = polls)
 *   N kernel nap (lane = part, a = 1 kicked / 0 timer / 2 plain timer, b = wait_us, c = owned wid mask)
 *   W hooked PPU task wait (TASKWAIT_IF2=1 [+ TASKWAIT_IF2_SECOND]; wid = workload whose arg the waited object holds,
 *     16 = none; lane = r29 side arg; a = lr, b = caller (saved LR), c = object EA)
 *   F guest flip request seen (a = flip count)
 * A helper thread writes qc_tl_<n>.txt (working directory) after each window. Parser: infamous2-recomp/_xfer/QC/tl.py.
 * Measurement only: no scheduling decision reads any of this. */
typedef struct { long long t0, t1; u32 a, b, c; u16 type; u8 wid, lane; } QcEv;
#define QC_N (1 << 21)   /* KP: was 1 << 19 (the U events of a 3 s window filled it) */
static QcEv* s_qc_ev;
static volatile long s_qc_head;
static volatile int g_qc_rec;
static double s_qc_win[8]; static int s_qc_nwin, s_qc_ms = 3000;
static LONGLONG s_qc_tbase, s_qc_f;
static volatile unsigned s_qc_lastflip;
static inline LONGLONG qc_now(void) { LARGE_INTEGER q; QueryPerformanceCounter(&q); return q.QuadPart; }
static void qc_ev(int type, u32 wid, u32 lane, LONGLONG t0, LONGLONG t1, u32 a, u32 b, u32 c)
{
    if (!g_qc_rec || !s_qc_ev) return;
    extern unsigned cellGcm_flip_request_count(void);
    const unsigned fl = cellGcm_flip_request_count();
    if (fl != s_qc_lastflip) {
        s_qc_lastflip = fl;
        const long j = InterlockedIncrement(&s_qc_head) - 1;
        if (j < QC_N) { QcEv* f = &s_qc_ev[j]; f->t0 = f->t1 = qc_now(); f->a = fl; f->b = f->c = 0; f->type = 'F'; f->wid = 0; f->lane = 0; }
    }
    const long i = InterlockedIncrement(&s_qc_head) - 1;
    if (i >= QC_N) return;
    QcEv* e = &s_qc_ev[i];
    e->t0 = t0; e->t1 = t1; e->a = a; e->b = b; e->c = c; e->type = (u16)type; e->wid = (u8)wid; e->lane = (u8)lane;
}
static void qc_dump(int n, double fz0, LONGLONG tend)
{
    char path[64]; snprintf(path, sizeof path, "qc_tl_%d.txt", n);
    FILE* o = fopen(path, "w");
    if (!o) { fprintf(stderr, "[QCTL] cannot write %s\n", path); return; }
    long cnt = s_qc_head; if (cnt > QC_N) cnt = QC_N;
    fprintf(o, "# QC_TL window %d: start fz %.3f s, %.1f ms, %ld events%s; columns: type wid lane t0_us dur_us a b c\n", n, fz0,
            1000.0 * (double)(tend - s_qc_tbase) / (double)s_qc_f, cnt, s_qc_head > QC_N ? " (BUFFER FULL)" : "");
    for (int w = 0; w < 16; w++) if (s_workloads[w].in_use) {
        const u64 a = vm_read64(s_workloads[w].spurs_ea + SPURS_WKL_INFO1 + (u32)w * SPURS_WKL_INFO_SZ + 8);
        fprintf(o, "# wkl %d ea %08X arg %016llX\n", w, s_workloads[w].spurs_ea, (unsigned long long)a);
    }
    for (long i = 0; i < cnt; i++) {
        const QcEv* e = &s_qc_ev[i];
        fprintf(o, "%c %u %u %.1f %.1f %u %u %08X\n", (char)e->type, e->wid, e->lane,
                1e6 * (double)(e->t0 - s_qc_tbase) / (double)s_qc_f, 1e6 * (double)(e->t1 - e->t0) / (double)s_qc_f, e->a, e->b, e->c);
    }
    fclose(o);
    fprintf(stderr, "[QCTL] wrote %s (%ld events)\n", path, cnt);
}
static DWORD WINAPI qc_thread(LPVOID p)
{
    (void)p;
    extern double ps3_fz_t(void);
    for (int w = 0; w < s_qc_nwin; w++) {
        while (ps3_fz_t() < s_qc_win[w]) Sleep(50);
        const double fz0 = ps3_fz_t();
        s_qc_head = 0; s_qc_lastflip = 0; s_qc_tbase = qc_now();
        InterlockedExchange((volatile long*)&g_qc_rec, 1);
        fprintf(stderr, "[QCTL] window %d recording at fz %.3f s for %d ms\n", w + 1, fz0, s_qc_ms);
        Sleep((DWORD)s_qc_ms);
        InterlockedExchange((volatile long*)&g_qc_rec, 0);
        const LONGLONG tend = qc_now();
        Sleep(100);   /* events already past the g_qc_rec check land */
        qc_dump(w + 1, fz0, tend);
    }
    return 0;
}
static void qc_init(void)
{
    static volatile long once = 0;
    if (once || InterlockedExchange(&once, 1)) return;
    const char* e = getenv("QC_TL");
    if (!e || !*e) return;
    for (const char* q = e; *q && s_qc_nwin < 8; ) {
        while (*q == ',' || *q == ' ') q++;
        if (!*q) break;
        s_qc_win[s_qc_nwin++] = atof(q);
        while (*q && *q != ',') q++;
    }
    { const char* m = getenv("QC_TL_MS"); if (m) s_qc_ms = atoi(m); if (s_qc_ms < 100) s_qc_ms = 100; if (s_qc_ms > 20000) s_qc_ms = 20000; }
    { LARGE_INTEGER f; QueryPerformanceFrequency(&f); s_qc_f = f.QuadPart; }
    s_qc_ev = (QcEv*)calloc(QC_N, sizeof(QcEv));
    if (!s_qc_ev || !s_qc_nwin) { fprintf(stderr, "[QCTL] off (bad QC_TL or no memory)\n"); return; }
    HANDLE h = CreateThread(NULL, 0, qc_thread, NULL, 0, NULL);
    if (h) CloseHandle(h);
    fprintf(stderr, "[QCTL] QC timeline: %d window(s) of %d ms, first at %.1f s -> qc_tl_<n>.txt\n", s_qc_nwin, s_qc_ms, s_qc_win[0]);
}
/* KP (2026-10-19): QC_TL events from other modules (sys_timer.c 'U' usleep poll runs, cellGcmSys.c 'A' RSX semaphore
 * acquire waits / 'P' ref publications). t0/t1 are QueryPerformanceCounter ticks. Measurement only. */
int ps3_qc_rec(void) { return g_qc_rec; }
void ps3_qc_ext(int type, uint32_t wid, uint32_t lane, long long t0, long long t1, uint32_t a, uint32_t b, uint32_t c)
{
    qc_ev(type, wid, lane, (LONGLONG)t0, (LONGLONG)t1, a, b, c);
}
/* v97f SPURS_LANE_CHAIN (default 1; 0 = off; off in PS3_AB phase 1): a lane whose policy-module run
 * found work runs the module AGAIN right away (same lane, same persistent context) instead of
 * waiting for the whole pass to join and the kernel to start the next one. play96 city: w1
 * (m_joblistMesh) needs ~22 passes per CTaskMgr wait; each lane is busy ~183 us per pass but the
 * pass lasts ~295 us (slowest lane + join + wake), and a lane claims ~1 job per run (4.5 GETLLARs
 * per busy run). A real SPU re-enters the module as soon as it exits, so an extra run is an
 * ordinary SPU schedule: same module, same lane, nothing else changes. Stops on the first run
 * that finds nothing (unless another lane finished a busy run meanwhile, max 2 such re-checks),
 * after SPURS_LANE_CHAIN_US (default 2000) of chaining, 64 re-runs, or when the workload is no
 * longer enabled/running or its argument changed. */
static int spurs_lane_chain_us(void)
{
    static int us = -2;
    if (us == -2) {
        const char* e = getenv("SPURS_LANE_CHAIN");
        const char* u = getenv("SPURS_LANE_CHAIN_US");
        int v = u ? atoi(u) : 2000;
        if (e && *e == '0') v = 0;
        if (v < 0) v = 0; if (v > 20000) v = 20000;
        fprintf(stderr, "[spurs-kern] v97f SPURS_LANE_CHAIN %s (busy lanes re-run at once, <= %d us per pass; off in PS3_AB phase 1)\n", v ? "ON" : "off", v);
        us = v;
    }
    if (g_abx & 0x4000) {   /* NB16 NB_CHAIN_B_US (default 6000): lane-chain budget per pass while ABX 0x4000 */
        static int b = -2; if (b == -2) { const char* e = getenv("NB_CHAIN_B_US"); b = e ? atoi(e) : 6000; if (b < 0) b = 0; if (b > 20000) b = 20000; }
        return v97_ab_off() ? 0 : b;
    }
    return v97_ab_off() ? 0 : us;
}
static int spurs_wkl_still_runs(u32 wid, u32 ea, uint64_t arg)
{
    if (wid >= 16 || !s_workloads[wid].in_use || s_workloads[wid].spurs_ea != ea) return 0;
    if (!(vm_read32(ea + SPURS_WKL_ENABLED) & (0x80000000u >> wid))) return 0;
    if (*(vm_base + ea + SPURS_WKL_STATE1 + wid) != 2) return 0;
    return vm_read64(ea + SPURS_WKL_INFO1 + wid * SPURS_WKL_INFO_SZ + 8) == arg;
}
/* Run one lane's policy module, chained while it keeps finding work. g_spurs_pm_polls = the sum. */
static unsigned spurs_lane_run_chain(struct LanePool* lp, spu_lifted_entry_fn fn, int image_id, const uint8_t* pm,
                                     uint32_t sz, uint64_t arg, u32 wid, u32 ea, u32 lane)
{
    extern SPU_PM_TLS volatile unsigned g_spurs_pm_polls;
    const int cus = lp ? spurs_lane_chain_us() : 0;
    /* v132 ABX 0x02 "hot idle lanes": a lane whose run found nothing does NOT leave the round while another lane of
     * the pool is still inside its chain; it waits (cv_hot) until some lane finishes a BUSY run (its dependents may
     * now be runnable) and runs the module again. The round ends when every lane is idle at the same time -- what
     * real SPUs do (they re-enter the module at once). Before: idle lanes parked until the slowest lane, the join and
     * the next kernel round (~260-290 us per round, ~12-25 rounds per slow PPU wait). Scheduling only. */
    const int hot = cus && (g_abx & 0x200);   /* v137: hot lanes moved off ABX 0x02 (measured ~0; 0x02 = rc plugins now) */
    if (hot) InterlockedIncrement(&lp->hot_active);
    long e = cus ? lp->ch_epoch : 0;
    const LONGLONG qc_c0 = g_qc_rec ? qc_now() : 0;   /* QC_TL */
    unsigned qc_runs = 1;
    spu_run_policy_module(fn, image_id, pm, sz, arg, wid, ea, lane);
    unsigned sum = g_spurs_pm_polls;
    if (qc_c0) qc_ev('M', wid, lane, qc_c0, qc_now(), sum, 0, 0);
    if (!cus) { if (qc_c0) qc_ev('C', wid, lane, qc_c0, qc_now(), qc_runs, sum, 0); return sum; }
    static LONGLONG s_f = 0;
    if (!s_f) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); s_f = f.QuadPart; }
    LARGE_INTEGER q; QueryPerformanceCounter(&q);
    const LONGLONG dl = q.QuadPart + (LONGLONG)cus * s_f / 1000000;
    unsigned p = sum, idle_re = 0;
    for (unsigned k = 0; k < (hot ? 1024u : 64u); k++) {
        if (p) {
            InterlockedIncrement(&lp->ch_epoch);
            if (hot) { AcquireSRWLockExclusive(&lp->lk); WakeAllConditionVariable(&lp->cv_hot); ReleaseSRWLockExclusive(&lp->lk); }
        } else if (hot) {
            /* wait under lp->lk: a busy finisher bumps ch_epoch BEFORE taking the lock to wake, an exiting lane drops
             * hot_active before it -> no lost wake-up. 1 ms timeout = safety net only. */
            /* keep waiting only while some lane of the round is still RUNNING (hot_idle < hot_active); when the last
             * runner goes idle too, everyone leaves -> the round ends exactly when all lanes are idle at once */
            int waited = 0;
            AcquireSRWLockExclusive(&lp->lk);
            lp->hot_idle++;
            if (lp->hot_idle >= lp->hot_active) WakeAllConditionVariable(&lp->cv_hot);
            while (lp->ch_epoch == e && lp->hot_idle < lp->hot_active) {
                QueryPerformanceCounter(&q);
                if (q.QuadPart >= dl) break;
                waited = 1;
                SleepConditionVariableSRW(&lp->cv_hot, &lp->lk, 1, 0);
            }
            lp->hot_idle--;
            ReleaseSRWLockExclusive(&lp->lk);
            if (spurs_stats_on() && waited) InterlockedIncrement(&lp->st_hot_wait);
            if (lp->ch_epoch == e) break;   /* everyone else left without new work (or deadline) */
            if (spurs_stats_on()) InterlockedIncrement(&lp->st_hot_rerun);
        }
        else if (lp->ch_epoch == e || idle_re >= 2) break;
        else idle_re++;
        QueryPerformanceCounter(&q);
        if (q.QuadPart >= dl || !spurs_wkl_still_runs(wid, ea, arg)) break;
        e = lp->ch_epoch;
        { const LONGLONG qc_m0 = qc_c0 ? qc_now() : 0;
        spu_run_policy_module(fn, image_id, pm, sz, arg, wid, ea, lane);
        p = g_spurs_pm_polls; sum += p;
        if (qc_m0) { qc_runs++; qc_ev('M', wid, lane, qc_m0, qc_now(), p, 0, 0); } }
        if (spurs_stats_on()) { InterlockedIncrement(&lp->st_ch_re); if (!p) InterlockedIncrement(&lp->st_ch_idle); }
    }
    if (hot) {
        AcquireSRWLockExclusive(&lp->lk); InterlockedDecrement(&lp->hot_active); WakeAllConditionVariable(&lp->cv_hot); ReleaseSRWLockExclusive(&lp->lk);
    }
    g_spurs_pm_polls = sum;
    if (qc_c0) qc_ev('C', wid, lane, qc_c0, qc_now(), qc_runs, sum, 0);   /* QC_TL */
    return sum;
}
/* v97f SPURS_PCORE_WIDS=<wid,...> (default "-" = none, EXPERIMENT knob; off in PS3_AB phase 1): the lane workers
 * (and the kernel thread while it runs the wid) of these workloads are restricted to the performance
 * cores (EfficiencyClass max) at ABOVE_NORMAL priority. v75's PS3_SPURS_PCORE pins wid 0 only; [TWCUR]
 * names the workloads the PPU really waits on (CTaskMgr w5..w10 in menus) -- their lanes may land on
 * the 12 E-core CPUs. 16 P-core CPUs fit ~2 more 5-lane workloads at most. Phase 1 restores the
 * default CPU set and NORMAL priority. No-op off Windows / on non-hybrid CPUs (priority only). */
static u32 spurs_pcore_mask(void)
{
    static int done = 0; static u32 m = 0;
    if (!done) {
        const char* e = getenv("SPURS_PCORE_WIDS");
        if (!e) e = "-";
        if (*e && *e != '-') { const char* q = e;
            while (*q) { while (*q == ',' || *q == ' ') q++; if (!*q) break;
                int w = atoi(q); if (w >= 0 && w < 16) m |= 1u << w;
                while (*q && *q != ',') q++; } }
        done = 1;
        fprintf(stderr, "[spurs-kern] v97f SPURS_PCORE_WIDS mask 0x%04X (lanes of these wids on P-cores, ABOVE_NORMAL; off in PS3_AB phase 1)\n", m);
    }
    return m;
}
static SPU_PM_TLS int t_v97_pc = 0;   /* 1 = this thread is currently pinned by v97f */
/* v126 SPURS_PCORE_WIDS_B=<wid,...> (default "-"): EXTRA wids whose lanes/kernel runs go to the P-cores ONLY while
 * g_ab_mask == SPURS_PCORE_B_PHASE (default 10 = 0b1010: no bit-mask consumer (bits 0/4/5) and no phase-index table
 * uses it). With PS3_AB=1 PS3_AB_N=... PS3_AB_SEQ=0,10 this is an in-run A/B of pinning the workloads behind the
 * HIDDEN main-thread job poll (func_00287058 -> func_000285A0 -> func_004015D8 -> task manager func_003F3A40;
 * ~20% of the main thread on heavy city routes, invisible to [TASKWAIT]). A thread pinned by the base mask is
 * never unpinned by the B rule (t_v97_pc 1 = pinned by base, 2 = pinned by B). [ABFPS] prints fps per phase. */
static u32 spurs_pcore_mask_b(void)
{
    static int done = 0; static u32 m = 0;
    if (!done) {
        const char* e = getenv("SPURS_PCORE_WIDS_B");
        if (!e) e = "-";
        if (*e && *e != '-') { const char* q = e;
            while (*q) { while (*q == ',' || *q == ' ') q++; if (!*q) break;
                int w = atoi(q); if (w >= 0 && w < 16) m |= 1u << w;
                while (*q && *q != ',') q++; } }
        done = 1;
        fprintf(stderr, "[spurs-kern] v126 SPURS_PCORE_WIDS_B mask 0x%04X (extra P-core wids only in PS3_AB phase %d)\n",
                m, getenv("SPURS_PCORE_B_PHASE") ? atoi(getenv("SPURS_PCORE_B_PHASE")) : 10);
    }
    return m;
}
static void spurs_v97_pcore_apply(u32 wid, const char* who, int want);
static void spurs_v97_pcore_sync(u32 wid, const char* who)
{
    if (wid >= 16) return;
    /* v133 ABX 0x20 (was 0x10 in v133a): w2 (BatchJobList -- the main thread's per-frame hidden job wait, ~75-80% SPU interpreter, lanes
     * unpinned so they can land on E-cores) runs its kernel/lane threads on the P-cores at ABOVE_NORMAL in that bit. */
    if (wid == 2 && !((spurs_pcore_mask() >> wid) & 1)) {
        const int want = (g_abx & 0x400) != 0;   /* v138: w2 pin moved off ABX 0x20 (measured ~0; 0x20 = audio lag) */
        if (want && t_v97_pc == 0) { t_v97_pc = 3; spurs_v97_pcore_apply(wid, who, 1); }
        else if (!want && t_v97_pc == 3) { t_v97_pc = 0; spurs_v97_pcore_apply(wid, who, 0); }
        return;
    }
    if ((spurs_pcore_mask() >> wid) & 1) {          /* v97f base rule (unchanged) */
        const int want = !v97_ab_off();
        if (want == (t_v97_pc != 0)) return;
        t_v97_pc = want;
        spurs_v97_pcore_apply(wid, who, want);
        return;
    }
    if ((spurs_pcore_mask_b() >> wid) & 1) {        /* v126 phase-B rule */
        static int s_ph = -1;
        if (s_ph < 0) { const char* e = getenv("SPURS_PCORE_B_PHASE"); s_ph = e ? atoi(e) : 10; }
        extern volatile int g_ab_mask;
        const int want = (g_ab_mask == s_ph);
        if (want && t_v97_pc == 0) { t_v97_pc = 2; spurs_v97_pcore_apply(wid, who, 1); }
        else if (!want && t_v97_pc == 2) { t_v97_pc = 0; spurs_v97_pcore_apply(wid, who, 0); }
    }
}
static void spurs_v97_pcore_apply(u32 wid, const char* who, int want)
{
#ifdef _WIN32
    { extern int ps3_place_hook(const char*, int, int); char nm[32];   /* HP2 PS3_PLACEMENT (default off) */
      snprintf(nm, sizeof nm, "w%u %s", (unsigned)wid, who);
      if (ps3_place_hook(nm, want, want ? THREAD_PRIORITY_ABOVE_NORMAL : THREAD_PRIORITY_NORMAL)) return; }
    typedef BOOL (WINAPI *GSCSI)(PSYSTEM_CPU_SET_INFORMATION, ULONG, PULONG, HANDLE, ULONG);
    typedef BOOL (WINAPI *STSCS)(HANDLE, const ULONG*, ULONG);
    static ULONG s_ids[256]; static volatile long s_n = -1;
    HMODULE k = GetModuleHandleA("kernel32.dll");
    STSCS ss = k ? (STSCS)GetProcAddress(k, "SetThreadSelectedCpuSets") : NULL;
    if (s_n < 0) {
        GSCSI gs = k ? (GSCSI)GetProcAddress(k, "GetSystemCpuSetInformation") : NULL;
        long n = 0;
        if (gs) {
            ULONG len = 0; gs(NULL, 0, &len, GetCurrentProcess(), 0);
            unsigned char* buf = (unsigned char*)malloc(len ? len : 1);
            if (buf && gs((PSYSTEM_CPU_SET_INFORMATION)buf, len, &len, GetCurrentProcess(), 0)) {
                BYTE maxc = 0, minc = 255;
                for (ULONG o = 0; o < len; ) { PSYSTEM_CPU_SET_INFORMATION c = (PSYSTEM_CPU_SET_INFORMATION)(buf + o);
                    if (c->Type == CpuSetInformation) { BYTE ec = c->CpuSet.EfficiencyClass; if (ec > maxc) maxc = ec; if (ec < minc) minc = ec; }
                    o += c->Size; }
                if (maxc != minc)
                    for (ULONG o = 0; o < len; ) { PSYSTEM_CPU_SET_INFORMATION c = (PSYSTEM_CPU_SET_INFORMATION)(buf + o);
                        if (c->Type == CpuSetInformation && c->CpuSet.EfficiencyClass == maxc && n < 256) s_ids[n++] = c->CpuSet.Id;
                        o += c->Size; }
            }
            free(buf);
        }
        s_n = n;
    }
    if (ss && s_n > 0) { if (want) ss(GetCurrentThread(), s_ids, (ULONG)s_n); else ss(GetCurrentThread(), NULL, 0); }
    /* No heterogeneous P-core set means there is no affinity benefit. Do not
     * boost every SPURS worker above the main/render threads on old low-core
     * CPUs; still restore NORMAL when an earlier pin is removed. */
    if (!want || s_n > 0)
        SetThreadPriority(GetCurrentThread(), want ? THREAD_PRIORITY_ABOVE_NORMAL : THREAD_PRIORITY_NORMAL);
    static volatile long s_logged = 0;
    if (s_logged < 24) { s_logged++;
        fprintf(stderr, "[PCORE] v97f %s w%u hosttid %lu: %s (%ld P-core CPUs)\n", who, wid, (unsigned long)GetCurrentThreadId(),
                want && s_n > 0 ? "-> P-cores, ABOVE_NORMAL" : "-> all CPUs, NORMAL", (long)s_n); }
#else
    (void)who; (void)wid; (void)want;
#endif
}
struct LaneWorkerArg { struct LanePool* lp; u32 idx; };
static DWORD WINAPI spurs_lane_worker(LPVOID p)
{
#ifdef _WIN32
    { ULONG g = 256 * 1024; SetThreadStackGuarantee(&g); }
#endif
    { extern void ps3_prof_register_spu_thread(void); ps3_prof_register_spu_thread(); }   /* v63 PS3_PROF_MAIN */
    struct LaneWorkerArg* wa = (struct LaneWorkerArg*)p;
    struct LanePool* lp = wa->lp;
    long my = 0;
    int last_busy = 0;   /* v96a: last lane run found work -> spin for the next pass first */
#ifdef _WIN32
    int pcore_applied = 0;
#endif
    for (;;) {
        struct spurs_pm_worker_arg j;
        if (last_busy && lp->gen == my) { int su = spurs_lane_spin_us(); if (su) spurs_lane_spin(&lp->gen, my, 1, su); }
        /* v132 ABX 0x04: while the PPU waits on THIS workload, its lanes stay hot (spin) instead of parking */
        while (lp->gen == my && spurs_tw_boost_wid(lp->job.wid)) spurs_lane_spin(&lp->gen, my, 1, 50);
        AcquireSRWLockExclusive(&lp->lk);
        while (lp->gen == my) SleepConditionVariableSRW(&lp->cv_go, &lp->lk, INFINITE, 0);
        my = lp->gen; j = lp->job;
        unsigned long long kt = lp->kick_tsc;
        ReleaseSRWLockExclusive(&lp->lk);
        if (kt && spu_laneprof_on() && j.wid < 16) {   /* v96a: gen++ -> this worker running */
            unsigned long long d = __rdtsc() - kt;
            __atomic_add_fetch(&g_lp_pool[j.wid][0], (long long)d, __ATOMIC_RELAXED);
            __atomic_add_fetch(&g_lp_pool[j.wid][1], 1, __ATOMIC_RELAXED);
        }
#ifdef _WIN32
        /* The PPU blocks on the main CTaskMgr (wid 0) every frame.  Its lane
         * helpers used to run at normal priority on any logical CPU, including
         * the E-cores of a hybrid Intel CPU, while the waiting PPU occupied a
         * P-core.  Keep only this latency-critical pool on P-cores; background
         * streaming/audio workloads remain free to use the E-cores.  Opt-in so
         * machines with few P-cores can retain the old scheduler policy. */
        if (!pcore_applied && j.wid == 0) {
            const char* e = getenv("PS3_SPURS_PCORE");
            if (e && *e && *e != '0') {
                extern void ps3_thread_prefer_pcores(const char*, int);
                ps3_thread_prefer_pcores("SPURS w0 lane", THREAD_PRIORITY_ABOVE_NORMAL);
            }
            pcore_applied = 1;
        }
        /* v132 ABX 0x01: w0 (never waited on by the main thread in [TWCUR]) leaves the P-cores in that bit, freeing
         * room for the waited workloads (w1/w6 lanes, main, FIFO). 18 threads >= ABOVE_NORMAL shared 16 P-core CPUs. */
        if (j.wid == 0) {
            static SPU_PM_TLS int t_w0 = -1;
            if (t_w0 < 0) { const char* e = getenv("PS3_SPURS_PCORE"); t_w0 = (e && *e && *e != '0') ? 1 : 2; }
            if (t_w0 != 2) { const int want = !(g_abx & 0x40);
                if (want != t_w0) { extern void ps3_thread_pcore_set(const char*, int, int);
                    t_w0 = want; ps3_thread_pcore_set("SPURS w0 lane", want, THREAD_PRIORITY_ABOVE_NORMAL); } }
        }
#endif
        spurs_v97_pcore_sync(j.wid, "lane");   /* v97f */
        extern SPU_PM_TLS volatile unsigned g_spurs_pm_polls;
        const int stats = spurs_stats_on();
        unsigned long long sp0 = stats ? g_spu_spin_polls : 0;
        LARGE_INTEGER q0, q1; if (stats) QueryPerformanceCounter(&q0);
        /* v31: j.spu_num = the lane the KERNEL thread runs (0 for a normal parallel
         * run, the rotate lane for a SPURS_PAR_KICK run); workers take the others. */
        u32 lane = (j.spu_num + 1 + wa->idx) % (lp->nworkers + 1);
        spurs_lane_run_chain(lp, j.fn, j.image_id, j.pm, j.pm_size, j.arg, j.wid, j.ea, lane);   /* v97f: was one run */
        if (stats) QueryPerformanceCounter(&q1);
        last_busy = g_spurs_pm_polls != 0;
        AcquireSRWLockExclusive(&lp->lk);
        lp->polls[wa->idx] = g_spurs_pm_polls;
        lp->spin[wa->idx] = stats ? g_spu_spin_polls - sp0 : 0;
        if (stats) {
            lp->st_ms_q[lane] += q1.QuadPart - q0.QuadPart;
            if (g_spurs_pm_polls) lp->st_busy[lane]++;
        }
        lp->done++;
        /* One kernel waits for the whole lane batch. Intermediate completions
         * cannot satisfy it and used to cause three needless wake/park cycles. */
        if ((u32)lp->done == lp->nworkers) WakeAllConditionVariable(&lp->cv_done);
        ReleaseSRWLockExclusive(&lp->lk);
    }
    return 0;
}
static int spurs_par_lanes_wid(u32 wid)
{
    static int s_init = 0; static u32 s_mask = 0;
    if (!s_init) {
        const char* e = getenv("SPURS_PAR_LANES");
        if (e) { const char* q = e;
            while (*q) { while (*q == ',' || *q == ' ') q++; if (!*q) break;
                int w = atoi(q); if (w >= 0 && w < 16) s_mask |= 1u << w;
                while (*q && *q != ',') q++; } }
        if (s_mask) fprintf(stderr, "[spurs-kern] SPURS_PAR_LANES mask 0x%04X (EXPERIMENTAL concurrent lanes)\n", s_mask);
        s_init = 1;
    }
    return wid < 16 && ((s_mask >> wid) & 1);
}
/* maxContention limits simultaneous execution, not the number of virtual SPU
 * identities that must visit the list. The experimental lane mask used to
 * override that limit implicitly, including via the idle-lane kick callback.
 * WWS atomic ticket claims do not make the jobs themselves safe to overlap.
 * Keep visiting all lanes serially unless the workload permits the entire
 * parallel sweep when the contention experiment is enabled. v104 gameplay
 * reproduced the freeze with serial lanes and lost substantial performance;
 * restore the established parallel default. SPURS_LANE_OVERCOMMIT=0 retains
 * the serial experiment for diagnosis, not as a movement fix. */
static int spurs_parallel_sweep_allowed(u32 wid, u32 lanes, u32 maxcont)
{
    if (lanes <= 1 || !spurs_par_lanes_wid(wid)) return 0;
    if (lanes <= maxcont) return 1;
    static SPU_PM_TLS int overcommit = -1;
    if (overcommit < 0) {
        const char* e = getenv("SPURS_LANE_OVERCOMMIT");
        overcommit = !(e && strcmp(e, "0") == 0);
    }
    /* Only the owning kernel calls this for a given wid. */
    static u8 reported[CELL_SPURS_MAX_WORKLOAD];
    if (wid < CELL_SPURS_MAX_WORKLOAD && !reported[wid]) {
        reported[wid] = 1;
        fprintf(stderr, "[SPURS-CONTENTION] w%u: %u virtual lanes, maxContention=%u; %s\n",
                wid, lanes, maxcont, overcommit
                    ? "parallel lane mode (v104 serial experiment disabled)"
                    : "serial lane visits (guest contention limit enforced)");
    }
    return overcommit;
}
/* Get (create on first use) the persistent lane pool of a workload. Only the
 * workload's owning kernel thread calls this. */
static struct LanePool* spurs_lane_pool(u32 wid, u32 nl)
{
    static struct LanePool* s_lp[16];
    struct LanePool* lp = s_lp[wid];
    if (!lp) {
        lp = (struct LanePool*)calloc(1, sizeof *lp);
        InitializeSRWLock(&lp->lk); InitializeConditionVariable(&lp->cv_go); InitializeConditionVariable(&lp->cv_done);
        lp->nworkers = nl - 1;
        for (u32 i = 0; i < lp->nworkers; i++) {
            struct LaneWorkerArg* wa = (struct LaneWorkerArg*)malloc(sizeof *wa);
            wa->lp = lp; wa->idx = i;
            CreateThread(NULL, 1u << 20, spurs_lane_worker, wa, 0, NULL);
        }
        s_lp[wid] = lp; s_lp_all[wid] = lp;
        fprintf(stderr, "[spurs-kern] wid %u: %u persistent lane workers\n", wid, lp->nworkers);
    }
    return lp;
}

/* v31 SPURS_PAR_KICK (default on; =0 off). v30f cutscene profile: w0's kernel thread
 * 84-90% busy, max run 104-140 ms, yet its parallel lanes sat at ~1% each with ~30 busy
 * runs per 10 s. Cause: SPURS_IDLE_ROTATE. After two empty sweeps an idle workload
 * visits ONE lane per pass, so the frame's CTaskMgr batch was always discovered by a
 * single rotate lane that then drained the WHOLE batch alone; the parallel sweep only
 * ran on the next pass and found nothing. Now a rotate run of a PAR workload arms a
 * one-shot hook (g_spurs_pm_poll_hook, fired on the module's first selectWorkload poll
 * = it found work): the hook wakes the lane workers for the OTHER lanes, and the run
 * joins them afterwards. Idle rotate runs never poll, so idle cost is unchanged. */
static SPU_PM_TLS struct LanePool* t_kick_lp;
static SPU_PM_TLS int t_kicked;
static void spurs_par_kick_cb(void)
{
    struct LanePool* lp = t_kick_lp;
    if (!lp || t_kicked) return;
    AcquireSRWLockExclusive(&lp->lk);
    lp->done = 0; lp->gen++; lp->kick_tsc = __rdtsc();
    WakeAllConditionVariable(&lp->cv_go);
    ReleaseSRWLockExclusive(&lp->lk);
    t_kicked = 1;
}
static int spurs_par_kick_on(void)
{
    static int s = -1;
    if (s < 0) { const char* e = getenv("SPURS_PAR_KICK"); s = (e && *e == '0') ? 0 : 1;
        fprintf(stderr, "[spurs-kern] SPURS_PAR_KICK %s (v31: rotate lane wakes the other lanes on first poll)\n", s ? "ON" : "off"); }
    return s;
}
/* Run ONE lane (sn) of a PAR workload on this thread; if it finds work, the other
 * lanes join in (see above). Returns the summed polls. */
static unsigned spurs_run_lane_kickable(u32 wid, u32 nl, u32 sn, spu_lifted_entry_fn fn, int image_id,
                                        const uint8_t* pm, uint32_t sz, uint64_t arg, u32 ea)
{
    extern SPU_PM_TLS volatile unsigned g_spurs_pm_polls;
    extern SPU_PM_TLS void (*g_spurs_pm_poll_hook)(void);
    struct LanePool* lp = spurs_lane_pool(wid, nl);
    const LONGLONG qc_q0 = g_qc_rec ? qc_now() : 0;   /* QC_TL */
    spurs_v97_pcore_sync(wid, "kernel");   /* v97f */
    AcquireSRWLockExclusive(&lp->lk);
    lp->job.fn = fn; lp->job.image_id = image_id; lp->job.pm = pm; lp->job.pm_size = sz;
    lp->job.arg = arg; lp->job.wid = wid; lp->job.ea = ea; lp->job.spu_num = sn;
    ReleaseSRWLockExclusive(&lp->lk);
    t_kick_lp = lp; t_kicked = 0;
    g_spurs_pm_poll_hook = spurs_par_kick_cb;
    spurs_lane_run_chain(lp, fn, image_id, pm, sz, arg, wid, ea, sn);   /* v97f: was one run */
    g_spurs_pm_poll_hook = 0;
    unsigned sum = g_spurs_pm_polls;
    if (t_kicked) {
        if ((u32)lp->done < lp->nworkers) { int su = spurs_lane_spin_us(); if (su) spurs_lane_spin(&lp->done, (long)lp->nworkers, 0, su); }   /* v96a */
        AcquireSRWLockExclusive(&lp->lk);
        while ((u32)lp->done < lp->nworkers) SleepConditionVariableSRW(&lp->cv_done, &lp->lk, INFINITE, 0);
        for (u32 i = 0; i < lp->nworkers; i++) { sum += lp->polls[i]; g_spu_spin_polls += lp->spin[i]; }
        if (spurs_stats_on()) lp->st_kicks++;
        ReleaseSRWLockExclusive(&lp->lk);
    }
    if (sum && spurs_stats_on()) lp->st_rot_busy++;
    if (qc_q0) qc_ev('Q', wid, 0, qc_q0, qc_now(), sum, (u32)t_kicked, sn);   /* QC_TL */
    t_kick_lp = NULL; t_kicked = 0;
    return sum;
}

/* Run lanes 0..nl-1 of one workload concurrently; returns the summed polls. */
static unsigned spurs_run_lanes_parallel(u32 wid, u32 nl, spu_lifted_entry_fn fn, int image_id,
                                         const uint8_t* pm, uint32_t sz, uint64_t arg, u32 ea)
{
    struct LanePool* lp = spurs_lane_pool(wid, nl);
    const LONGLONG qc_r0 = g_qc_rec ? qc_now() : 0;   /* QC_TL */
    spurs_v97_pcore_sync(wid, "kernel");   /* v97f */
    AcquireSRWLockExclusive(&lp->lk);
    lp->job.fn = fn; lp->job.image_id = image_id; lp->job.pm = pm; lp->job.pm_size = sz;
    lp->job.arg = arg; lp->job.wid = wid; lp->job.ea = ea; lp->job.spu_num = 0;
    lp->done = 0; lp->gen++; lp->kick_tsc = __rdtsc();
    WakeAllConditionVariable(&lp->cv_go);
    ReleaseSRWLockExclusive(&lp->lk);
    extern SPU_PM_TLS volatile unsigned g_spurs_pm_polls;
    const int stats = spurs_stats_on();
    LARGE_INTEGER q0, q1; if (stats) QueryPerformanceCounter(&q0);
    spurs_lane_run_chain(lp, fn, image_id, pm, sz, arg, wid, ea, 0);   /* v97f: was one run */
    if (stats) QueryPerformanceCounter(&q1);
    unsigned sum = g_spurs_pm_polls;
    unsigned long long jw0 = spu_laneprof_on() ? __rdtsc() : 0;   /* v96a: kernel lane done -> all lanes done */
    if ((u32)lp->done < lp->nworkers) { int su = spurs_lane_spin_us(); if (su) spurs_lane_spin(&lp->done, (long)lp->nworkers, 0, su); }   /* v96a */
    AcquireSRWLockExclusive(&lp->lk);
    if (stats) {
        lp->st_ms_q[0] += q1.QuadPart - q0.QuadPart;
        if (g_spurs_pm_polls) lp->st_busy[0]++;
        lp->st_runs++;
    }
    while ((u32)lp->done < lp->nworkers) SleepConditionVariableSRW(&lp->cv_done, &lp->lk, INFINITE, 0);
    for (u32 i = 0; i < lp->nworkers; i++) { sum += lp->polls[i]; g_spu_spin_polls += lp->spin[i]; }
    ReleaseSRWLockExclusive(&lp->lk);
    if (jw0 && wid < 16) {
        __atomic_add_fetch(&g_lp_pool[wid][2], (long long)(__rdtsc() - jw0), __ATOMIC_RELAXED);
        __atomic_add_fetch(&g_lp_pool[wid][3], 1, __ATOMIC_RELAXED);
    }
    if (qc_r0) qc_ev('R', wid, 0, qc_r0, qc_now(), sum, (u32)arg, nl);   /* QC_TL */
    return sum;
}
/* v30c: one line per parallel-lane workload, printed with [pm-time] (resets the window). */
static void spurs_lanes_report(double win)
{
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    for (int w = 0; w < 16; w++) {
        struct LanePool* lp = s_lp_all[w];
        if (!lp || (!lp->st_runs && !lp->st_kicks && !lp->st_rot_busy)) continue;
        fprintf(stderr, "[pm-lanes] w%d %ld parallel runs, %ld/%ld busy rotate runs kicked (v31):", w, lp->st_runs,
                lp->st_kicks, lp->st_rot_busy);
        if (lp->st_ch_re) fprintf(stderr, " chain re-runs %ld (%ld empty) |", lp->st_ch_re, lp->st_ch_idle);   /* v97f */
        if (lp->st_hot_wait || lp->st_hot_rerun) fprintf(stderr, " hot waits %ld re-runs %ld (abx 0x%02X) |", lp->st_hot_wait, lp->st_hot_rerun, g_abx);   /* v132 */
        lp->st_hot_wait = lp->st_hot_rerun = 0;
        lp->st_ch_re = lp->st_ch_idle = 0;
        lp->st_kicks = 0; lp->st_rot_busy = 0;
        for (u32 k = 0; k <= lp->nworkers && k < 8; k++) {
            fprintf(stderr, " l%u %.0fms(%.0f%%)/%ldb", k, 1000.0 * lp->st_ms_q[k] / f.QuadPart,
                    100.0 * lp->st_ms_q[k] / f.QuadPart / win, lp->st_busy[k]);
            lp->st_ms_q[k] = 0; lp->st_busy[k] = 0;
        }
        lp->st_runs = 0;
        fprintf(stderr, "%c", 10);
    }
}

#ifndef _WIN32
#include <sched.h>
#include <unistd.h>
#endif
extern volatile int g_ab_mask;
/* v82 SPURS_KICK (default on; =0 off). After SPURS_SPIN_US without work a kernel thread waits
 * SPURS_IDLE_WAIT_US per empty pass, so a CTaskMgr job submitted during that window was only
 * discovered when the timer expired -- while the PPU sat in its usleep(1) completion poll
 * (func_002CFFE0, lr 002D0054, 20-47% of the main thread). Now an idle kernel thread also waits
 * on its own kick event, and any guest usleep <= SPURS_KICK_US (default 100, a job-completion
 * poll) wakes every kernel thread that is currently parked. Costs one InterlockedExchange per
 * poll when nobody is parked. */
#define SPURS_KICK_MAX 64
static HANDLE s_kick_ev[SPURS_KICK_MAX];
static volatile long s_kick_waiting[SPURS_KICK_MAX];
static volatile long s_kick_n = 0, s_kick_parked = 0;
static volatile long s_kick_woke = 0, s_kick_calls = 0;
static int spurs_kick_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("SPURS_KICK"); on = (e && *e == '1') ? 1 : 0;   /* v83: default OFF (play82 A/B: -1 fps) */
        fprintf(stderr, "[spurs-kern] SPURS_KICK %s (v82: guest short usleep wakes parked kernel threads)\n", on ? "ON" : "off"); }
    return on;
}
/* usleep threshold for the v82 kick; -1 = the usleep kick is off (v83 default). */
int ps3_spurs_kick_us(void)
{
    static int us = -2;
    if (us == -2) { const char* e = getenv("SPURS_KICK_US"); us = spurs_kick_on() ? (e ? atoi(e) : 100) : -1; }
    return us;
}

/* v83 TASKWAIT_BOOST (default on; =0 off). play82 PROF: ~40% of the main thread is func_0041FC38 spinning on
 * func_0041D028(taskset, 3, n) = "have ALL SPU lanes' progress cursors (u16 at slot+0x42..0x4E) reached n?".
 * Real SPUs poll the task list continuously, so the cursors move at once; our lanes only move them when the
 * kernel thread next runs a pass, and after 0.5 ms without work that thread naps 250 us per empty pass. The
 * generated func_0041FC38 now calls ps3_taskwait_begin/_spin/_end: begin wakes every parked kernel thread
 * once, and while a wait is open the kernel thread that owns CTaskMgr (instance 0 / part 0) keeps passing
 * instead of napping (phase 2 of the v83 PS3_AB cycle: every kernel thread does). [TASKWAIT] every 10 s. */
volatile long g_spurs_ppu_waiting = 0;
/* v92 TASKWAIT_SWEEP (default 1; 0 off): the main thread's CTaskMgr wait (func_0041D028) needs EVERY lane's
 * progress cursor to reach n. Once the list is idle the kernel visits ONE lane per pass (SPURS_IDLE_ROTATE)
 * and naps between passes, so the last cursors trail by up to nl passes -- ~1 ms mean, 10-15 ms max per wait
 * in play91 ([TASKWAIT] 30-34% of wall). While any such wait is open, idle workloads sweep all lanes. */
volatile long g_spurs_ppu_tw = 0;
volatile long long g_spurs_tw_open_us = 0;   /* v97b: start (us) of the open PPU CTaskMgr wait, 0 = none */
static int spurs_tw_sweep_on(void)
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("TASKWAIT_SWEEP"); on = (e && *e == '0') ? 0 : 1;
        fprintf(stderr, "[spurs-kern] v92 TASKWAIT_SWEEP %s (full lane sweeps while the PPU waits on a task list)\n", on ? "ON" : "off"); }
    return on;
}
static int spurs_taskwait_mode(void)   /* 0 off, 1 owner kernel only, 2 every kernel */
{
    static int on = -1;
    if (on < 0) { const char* e = getenv("TASKWAIT_BOOST"); on = (e && *e == '1') ? 1 : 0;   /* v84: default OFF (play83 A/B: no gain) */
        fprintf(stderr, "[spurs-kern] TASKWAIT_BOOST %s (v83: no napping while the PPU waits on CTaskMgr)\n", on ? "ON" : "off"); }
    /* v129 TASKWAIT_BOOST_PHASE=<phase> (default -1 = none): in that PS3_AB phase EVERY kernel keeps polling while
     * the PPU is in a hooked wait (mode 2), whatever TASKWAIT_BOOST says -> in-run A/B for the hidden job poll in
     * func_00287058 (now hooked), whose "done" needs every lane's progress cursor to move (v82 lesson). */
    { static int s_bp = -2; if (s_bp == -2) { const char* e = getenv("TASKWAIT_BOOST_PHASE"); s_bp = (e && *e) ? atoi(e) : -1; }
      if (s_bp >= 0 && g_ab_mask == s_bp) return 2; }
    if (!on || g_ab_mask == 1) return 0;
    return g_ab_mask == 2 ? 2 : 1;
}
/* v97b [STALL] guard for the v31d stall breaker. The breaker zeroed + reloaded every lane of ANY
 * workload that found no work for SPURS_STALL_MS -- in play96 that only happened to workloads that
 * simply HAD no work (all wids at once during 2 s main-thread hitches / zone switches, LibMP3 etc.);
 * the v31d case it was written for (unclaimed jobs, blind PM) was really the lost PPU decrement
 * fixed in v32c. v97b2: SPURS_STALL_MODE 1 (default) = restart only while the PPU has been stuck in a
 * CTaskMgr wait for > SPURS_STALL_MS (a real hang -- the only case the breaker ever existed for);
 * 0 = old v31d behaviour (restart any idle workload); 2 = never restart, log only; 3 = mode 1 OR the
 * workload's queue line (128 bytes at its workload data EA) changed since its last busy run (weak:
 * BatchJobList w2's line changes ~50x/s while it has nothing runnable). SPURS_STALL_GAP_MS (default 100; 0 off): log "[STALL] late pickup"
 * when published work waited that long before any lane found it. */
static unsigned char s_stl_line[MAX_SPURS_INST][16][128];
static u32 s_stl_ea[MAX_SPURS_INST][16];
static long long s_stl_pend_us[MAX_SPURS_INST][16];   /* first outside change of the line not yet followed by a busy run */
static long s_stl_chg[MAX_SPURS_INST][16];            /* outside changes since the last busy run */
static long long s_stl_last_us[MAX_SPURS_INST][16];   /* v103B: time of the LATEST outside change */
static long long s_stl_busy_us[MAX_SPURS_INST][16];   /* v103B: time of the last busy run */
static long long tw_now_us(void);
static double stl_t(void)   /* v97b2: seconds since the first [STALL] query (~boot) */
{
    static long long t0 = 0; long long n = tw_now_us();
    if (!t0) t0 = n;
    return (double)(n - t0) / 1e6;
}
/* v100: SPURS_STALL_MODE 4 (NEW DEFAULT) = restart only when the PPU task wait has been stuck for
 * > SPURS_STALL_HARD_MS (default 10000 ms) -- a genuine deadlock, never an ordinary hitch.
 * play99: the only gameplay hitch (1 s, t=319 s, everything blocked in atomics) made mode 1 restart
 * w2/w5/w6/w8/w9 at t=320.2 s = the user's T-pose time. A restart zeroes the persistent wwsjob PM LS
 * (claimed/staged job pipeline), so jobs it had already claimed never run -> the characters whose
 * animation jobs were in flight stay in the bind pose, frozen; nearby NPCs despawn. SPURS_STALL_MODE=1
 * restores v97b2. */
static long long s_stall_hard_us = 10000000LL;
static int spurs_stall_mode(void)
{
    static int m = -1;
    if (m < 0) { const char* e = getenv("SPURS_STALL_MODE"); m = (e && *e) ? atoi(e) : 4; if (m < 0 || m > 4) m = 4;
        { const char* h = getenv("SPURS_STALL_HARD_MS"); if (h && *h) { long v = atol(h); if (v > 0) s_stall_hard_us = (long long)v * 1000LL; } }
        fprintf(stderr, "[STALL] v100 SPURS_STALL_MODE=%d (%s), hard limit %lld ms\n", m,
                m == 0 ? "old: restart any workload idle > SPURS_STALL_MS" :
                m == 1 ? "restart only while a PPU task wait is stuck" :
                m == 2 ? "log only, never restart" :
                m == 3 ? "restart on a stuck PPU wait or a changed queue line" :
                         "restart only when a PPU task wait is stuck > SPURS_STALL_HARD_MS (v100 default)",
                s_stall_hard_us / 1000); }
    return m;
}
/* Before a workload's run: did anyone else change its queue line since our last run? */
static void spurs_stall_pre(int sl, u32 wid, u32 line_ea)
{
    if (sl < 0 || sl >= MAX_SPURS_INST || wid >= 16 || !line_ea) return;
    static int s_ii = 0; if (!s_ii) { s_ii = 1; (void)stl_t(); }
    const unsigned char* p = (const unsigned char*)vm_base + line_ea;
    if (s_stl_ea[sl][wid] != line_ea) {          /* new list: the PM context is reset anyway */
        s_stl_ea[sl][wid] = line_ea; memcpy(s_stl_line[sl][wid], p, 128);
        s_stl_pend_us[sl][wid] = 0; s_stl_chg[sl][wid] = 0;
        return;
    }
    if (memcmp(p, s_stl_line[sl][wid], 128) != 0) {
        const long long n = tw_now_us();
        if (!s_stl_pend_us[sl][wid]) s_stl_pend_us[sl][wid] = n;
        s_stl_last_us[sl][wid] = n;
        s_stl_chg[sl][wid]++;
        memcpy(s_stl_line[sl][wid], p, 128);   /* v103B: count changes, not runs that saw the same changed line */
    }
}
/* After the run (all lanes joined): late-pickup log, then snapshot what the run itself left. */
static void spurs_stall_post(int sl, u32 wid, int busy)
{
    if (sl < 0 || sl >= MAX_SPURS_INST || wid >= 16 || !s_stl_ea[sl][wid]) return;
    if (busy) {
        static int s_gap = -1; static long s_late_n = 0;
        if (s_gap < 0) { const char* e = getenv("SPURS_STALL_GAP_MS"); s_gap = e ? atoi(e) : 100; }
        long long p0 = s_stl_pend_us[sl][wid];
        if (s_gap > 0 && p0) {
            const long long now = tw_now_us(), d = now - p0;
            if (d > (long long)s_gap * 1000) {
                s_late_n++;
                /* v103B: + how long before the pickup the LAST outside change happened. A list whose header line
                 * changes all the time (BatchJobList w2: ~50x/s with nothing runnable) "waits" from its first
                 * change; if the last change was a few ms before the pickup, the work was picked up at once and
                 * nothing was missed. "last change >= the gap" = a published change that no lane acted on. */
                const long long dl = s_stl_last_us[sl][wid] ? now - s_stl_last_us[sl][wid] : -1;
                const int real = dl >= (long long)s_gap * 1000;
                if (s_late_n <= 200 || (s_late_n % 100) == 0 || real)
                    fprintf(stderr, "[STALL] t=%.1fs late pickup: inst %d wid %u found work %.0f ms after its queue line changed (%ld outside changes, last change %.1f ms before the pickup -> %s, PPU wait open %s, #%ld)\n",
                            stl_t(), sl, wid, d / 1000.0, s_stl_chg[sl][wid], dl / 1000.0,
                            real ? "MISSED: the queue line was unchanged that long before a lane saw work" : "picked up right after the latest change, nothing missed",
                            g_spurs_tw_open_us ? "yes" : "no", s_late_n);
            }
        }
        s_stl_pend_us[sl][wid] = 0; s_stl_chg[sl][wid] = 0;
        s_stl_busy_us[sl][wid] = tw_now_us();
    }
    memcpy(s_stl_line[sl][wid], (const unsigned char*)vm_base + s_stl_ea[sl][wid], 128);
}
/* v103B [FREEZE-SPURS]: the F8 dump (sys_ppu_thread.c fz_dump) also prints every SPURS workload: lifecycle
 * state, enabled bit, data EA, runs/busy since Add, ms since its last run / busy run, run flag (quantum in
 * flight), readyCount, outside queue-line changes not yet followed by a busy run, the head of its data
 * (CTaskMgr / WWS rows) and each lane's persistent policy-module context (pc, status, job pipeline words).
 * Names the workload/lane that stopped producing results when the characters freeze. */
static void spurs_freeze_dump(const char* why)
{
    extern void spurs_pm_lane_brief(uint32_t wid, char* b, int cap);   /* spurs_policy.c */
    const long long now = tw_now_us();
    fprintf(stderr, "[FREEZE-SPURS] ===== SPURS workloads (%s) t=%.1fs =====\n", why ? why : "?", stl_t());
    for (int sl = 0; sl < MAX_SPURS_INST; sl++) {
        const u32 ea = s_inst[sl].ea;
        if (!ea) continue;
        const u32 en = vm_read32(ea + SPURS_WKL_ENABLED);
        for (u32 w = 0; w < 16; w++) {
            if (!s_workloads[w].in_use || s_workloads[w].spurs_ea != ea) continue;
            const u64 dd = s_workloads[w].data;
            const u32 data = (u32)(dd >> 32) ? (u32)(dd >> 32) : (u32)dd;
            fprintf(stderr, "[FREEZE-SPURS] inst %d \"%s\" w%u: state %u enabled %u data 0x%08X gen %ld | %ld runs / %ld busy since Add, "
                            "last run %.1f ms ago, last busy run %.1f ms ago | run flag %ld | ready %u curcont %u | queue line: %ld outside changes "
                            "pending (first %.1f ms ago, last %.1f ms ago)\n",
                    sl, s_inst[sl].prefix, w, *(vm_base + ea + SPURS_WKL_STATE1 + w), (en >> (31 - w)) & 1, data, s_wkl_gen[w],
                    s_wkl_runs[w], s_wkl_busy_runs[w],
                    s_wkl_last_run_us[w] ? (now - s_wkl_last_run_us[w]) / 1000.0 : -1.0,
                    s_wkl_last_busy_us[w] ? (now - s_wkl_last_busy_us[w]) / 1000.0 : -1.0,
                    (long)s_wkl_run_flag[w], *(vm_base + ea + SPURS_WKL_READY1 + w), *(vm_base + ea + SPURS_WKL_CURCONT + w),
                    s_stl_chg[sl][w], s_stl_pend_us[sl][w] ? (now - s_stl_pend_us[sl][w]) / 1000.0 : -1.0,
                    s_stl_last_us[sl][w] ? (now - s_stl_last_us[sl][w]) / 1000.0 : -1.0);
            wkl_dump_line("f8", w, data);
            char b[1024]; b[0] = 0;
            spurs_pm_lane_brief(w, b, (int)sizeof b);
            if (b[0]) fprintf(stderr, "[FREEZE-SPURS]   w%u lanes:%s\n", w, b);
        }
    }
    fprintf(stderr, "[FREEZE-SPURS] ===== end =====\n");
}
void ps3_spurs_kick(void);
static SPU_PM_TLS long long t_tw_t0;
static SPU_PM_TLS int t_tw_marked;
static long long s_tw_n, s_tw_us, s_tw_max, s_tw_fast, s_tw_win0, s_tw_spins;
static long long tw_now_us(void)
{
    static LONGLONG f = 0; LARGE_INTEGER q;
    if (!f) { LARGE_INTEGER ff; QueryPerformanceFrequency(&ff); f = ff.QuadPart; }
    QueryPerformanceCounter(&q);
    return (long long)(q.QuadPart / f) * 1000000LL + (long long)(q.QuadPart % f) * 1000000LL / f;
}
/* v97f [TWCUR]: WHICH workload the PPU's CTaskMgr wait (func_0041FC38 -> func_0041D028) waits on.
 * The waited object's first word is the task list = the workload's data EA (wklInfo arg; the cursor
 * row polled is data+0x70). Each wait is attributed to the wid whose live arg matches; [TWCUR] prints
 * waits / ms / >=100 us waits / max per wid every 10 s with [TASKWAIT]. LANEPROF=0 -> off. */
#include "../../runtime/ppu/ppu_context.h"
extern PPU_THREAD_LOCAL ppu_context* g_active_ctx;
static SPU_PM_TLS int t_tw_wid = -1;
static long long s_tww_n[17], s_tww_us[17], s_tww_long[17], s_tww_max[17];
static int tw_wid_of_wait(void)
{
    ppu_context* c = g_active_ctx;
    if (!c) return 16;
    /* The wait-object register is title-specific: inFamous 2's
     * func_00371054 holds its job list in r28, not r30. */
    static int s_obj_reg = -1;
    if (s_obj_reg < 0) {
        const char* e = getenv("TASKWAIT_OBJ_REG");
        const int r = e ? atoi(e) : 30;
        s_obj_reg = (r >= 0 && r < 32) ? r : 30;
    }
    u32 obj = (u32)c->gpr[s_obj_reg];
    /* NB16 TASKWAIT_OBJ_REG2=<r> (default -1): try this register when the first does not name a live workload */
    static int s_obj_reg2 = -2;
    if (s_obj_reg2 == -2) { const char* e = getenv("TASKWAIT_OBJ_REG2"); const int r = e ? atoi(e) : -1; s_obj_reg2 = (r >= 0 && r < 32) ? r : -1; }
    if (s_obj_reg2 >= 0) {
        const u32 o2 = (u32)c->gpr[s_obj_reg2];
        int ok1 = 0;
        if (obj && obj < 0xF0000000u) { const u32 b1 = vm_read32(obj);
            for (int w = 0; w < 16 && !ok1; w++) if (s_workloads[w].in_use) {
                const u64 a = vm_read64(s_workloads[w].spurs_ea + SPURS_WKL_INFO1 + (u32)w * SPURS_WKL_INFO_SZ + 8);
                if (b1 && ((u32)a == b1 || (u32)(a >> 32) == b1)) ok1 = 1; } }
        if (!ok1 && o2 && o2 < 0xF0000000u) obj = o2;
    }
    if (!obj || obj >= 0xF0000000u) return 16;
    const u32 base = vm_read32(obj);
    static volatile u32 s_last_base = 0; static volatile int s_last_wid = 16;
    if (base && base == s_last_base) {   /* re-validate the cached hit (workloads get re-added) */
        const int w = s_last_wid;
        if (w < 16 && s_workloads[w].in_use) {
            const u64 a = vm_read64(s_workloads[w].spurs_ea + SPURS_WKL_INFO1 + (u32)w * SPURS_WKL_INFO_SZ + 8);
            if ((u32)a == base || (u32)(a >> 32) == base) return w;
        }
    }
    for (int w = 0; w < 16; w++) {
        if (!s_workloads[w].in_use) continue;
        const u64 a = vm_read64(s_workloads[w].spurs_ea + SPURS_WKL_INFO1 + (u32)w * SPURS_WKL_INFO_SZ + 8);
        if ((u32)a == base || (u32)(a >> 32) == base) { s_last_wid = w; s_last_base = base; return w; }
    }
    return 16;
}

/* ---- v113 FRAMEREC=1: per-50 ms timeline of the counters below, last ~12 s in a ring; F8 writes the
 * last ~6 s to framerec_N.txt ([FREEZE] time base). Question it answers: during a world-movement hitch
 * (sim keeps running, drawn positions freeze ~1 s), what changes slice by slice -- presents/draws,
 * CTaskMgr waits per workload, SPU busy per workload, FIOS reads / event-flag waits. Counters are
 * cumulative and never reset; the sampler only reads them. */
volatile long long g_fr_tw_us[17], g_fr_tw_n[17];
volatile long long g_fr_busy_us[2][16], g_fr_runs[2][16];
volatile long long g_fr_ef_us, g_fr_ef_n, g_fr_reads, g_fr_presents, g_fr_draws;
void (*volatile g_ps3_fr_dump)(void) = 0;   /* sys_ppu_thread.c F8 */
typedef struct { double t; long long tw_us[17], tw_n[17], busy[2][16], runs[2][16], ef_us, ef_n, reads, pres, draws; unsigned flips; } FrSnap;
#define FR_N 256
static FrSnap s_fr[FR_N];
static volatile long s_fr_head;
static DWORD WINAPI fr_thread(LPVOID p)
{
    (void)p;
    extern double ps3_fz_t(void);
    extern unsigned cellGcm_flip_request_count(void);
    for (;;) {
        Sleep(50);
        FrSnap* s = &s_fr[s_fr_head % FR_N];
        s->t = ps3_fz_t();
        for (int i = 0; i < 17; i++) { s->tw_us[i] = g_fr_tw_us[i]; s->tw_n[i] = g_fr_tw_n[i]; }
        for (int a = 0; a < 2; a++) for (int w = 0; w < 16; w++) { s->busy[a][w] = g_fr_busy_us[a][w]; s->runs[a][w] = g_fr_runs[a][w]; }
        s->ef_us = g_fr_ef_us; s->ef_n = g_fr_ef_n; s->reads = g_fr_reads; s->pres = g_fr_presents; s->draws = g_fr_draws;
        s->flips = cellGcm_flip_request_count();
        InterlockedIncrement(&s_fr_head);
    }
    return 0;
}

/* ---- v115 FTL_LINE=<hex EA of the FIOS event flag> (inFamous 405C4700): microsecond timeline of one
 * streaming request round trip. Events: 1 LFQueue push (PPU), 6/7 task WAIT_SIGNAL enter/exit (SPU task),
 * 3 DMA PUT / 4 PUTLLC|PUTLLUC commit / 5 spurs_ef_set_from_spu touching the flag line (SPU -> "done"),
 * 8/9 EventFlagWait enter/exit on that flag (PPU). Ring of 2^16; F8 (with FRAMEREC=1) writes ftl_N.txt
 * (last 3 s). v114 proved streaming speed sets the hitch length; this names the slow leg. */
volatile uint32_t g_ftl_line = 0xFFFFFFFFu;   /* resolved on first use */
typedef struct { long long t; uint32_t a, b; uint32_t tid; int type; } FtlEv;
#define FTL_N (1 << 16)
static FtlEv s_ftl[FTL_N];
static volatile long s_ftl_head;
static uint32_t ftl_line(void)
{
    if (g_ftl_line == 0xFFFFFFFFu) { const char* e = getenv("FTL_LINE"); g_ftl_line = e ? ((uint32_t)strtoul(e, 0, 16) & ~127u) : 0;
        if (g_ftl_line) fprintf(stderr, "[FTL] v115 request timeline on flag line 0x%08X (F8 -> ftl_N.txt)\n", g_ftl_line); }
    return g_ftl_line;
}
void ps3_ftl(int type, uint32_t a, uint32_t b)
{
    if (!ftl_line()) return;
    LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    long i = InterlockedIncrement(&s_ftl_head) - 1;
    FtlEv* e = &s_ftl[i & (FTL_N - 1)];
    e->t = 0;
    e->a = a; e->b = b; e->type = type; e->tid = (uint32_t)GetCurrentThreadId();
    e->t = (long long)(c.QuadPart / f.QuadPart) * 1000000LL + (long long)((c.QuadPart % f.QuadPart) * 1000000LL / f.QuadPart);
}
static void ftl_dump(int n)
{
    if (!ftl_line()) return;
    extern double ps3_fz_t(void);
    LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    const long long now = (long long)(c.QuadPart / f.QuadPart) * 1000000LL + (long long)((c.QuadPart % f.QuadPart) * 1000000LL / f.QuadPart);
    const double tnow = ps3_fz_t();
    char path[64]; snprintf(path, sizeof path, "ftl_%d.txt", n);
    FILE* o = fopen(path, "w"); if (!o) return;
    static const char* nm[] = {"?", "PUSH", "TASK_SEES", "DMA_PUT_FLAG", "ATOMIC_FLAG", "EF_SET_SPU", "TASK_SLEEP", "TASK_WAKE", "PPU_WAIT", "PPU_WOKE"};
    fprintf(o, "# %s: t(s, [FREEZE] base) event hosttid a b. PUSH a=queue; DMA_PUT_FLAG/ATOMIC_FLAG a=ea b=size|cmd;"
               " TASK_* a=taskset b=task; PPU_WAIT a=flag b=pattern; PPU_WOKE a=flag b=bits\n", path);
    long head = s_ftl_head, lo = head > FTL_N ? head - FTL_N : 0, cnt = 0;
    for (long i = lo; i < head; i++) {
        FtlEv e = s_ftl[i & (FTL_N - 1)];
        if (!e.t || now - e.t > 3000000) continue;
        fprintf(o, "%.6f %-12s %6u %08X %08X\n", tnow - (now - e.t) / 1e6, (e.type > 0 && e.type < 10) ? nm[e.type] : "?", e.tid, e.a, e.b);
        cnt++;
    }
    fclose(o);
    fprintf(stderr, "[FTL] wrote %s (%ld events)\n", path, cnt);
}
static void fr_dump(void)
{
    static int n = 0;
    if (n >= 12) return;
    ftl_dump(n + 1);   /* v115 */
    long head = s_fr_head;
    if (head < 3) return;
    long lo = head - 121; if (lo < 0) lo = 0; if (lo < head - (FR_N - 2)) lo = head - (FR_N - 2);
    char path[64]; snprintf(path, sizeof path, "framerec_%d.txt", ++n);
    FILE* f = fopen(path, "w"); if (!f) return;
    /* columns: only workloads with activity in the window */
    int twc[17] = {0}, bc[2][16] = {{0}};
    for (long i = lo + 1; i < head; i++) {
        const FrSnap* a = &s_fr[(i - 1) % FR_N]; const FrSnap* b = &s_fr[i % FR_N];
        for (int w = 0; w < 17; w++) if (b->tw_us[w] - a->tw_us[w] > 0) twc[w] = 1;
        for (int k = 0; k < 2; k++) for (int w = 0; w < 16; w++) if (b->busy[k][w] - a->busy[k][w] > 0) bc[k][w] = 1;
    }
    fprintf(f, "# %s: 50 ms slices before F8. dt_ms = slice length; pres/flips = frames presented / flips requested; "
               "draws/p = draws per presented frame; reads = cellFsRead; ef = EventFlagWait count/ms (FIOS);\n"
               "# tw:wN = ms the PPU spent in CTaskMgr waits on workload N (w16 = unmatched); bz:iK.wN = SPU busy ms of instance K workload N\n", path);
    fprintf(f, "%9s %6s %4s %5s %7s %5s %4s %6s %7s", "t", "dt_ms", "pres", "flips", "draws/p", "reads", "efN", "efMs", "twTotMs");
    for (int w = 0; w < 17; w++) if (twc[w]) fprintf(f, " %7s%-2d", "tw:w", w);
    for (int k = 0; k < 2; k++) for (int w = 0; w < 16; w++) if (bc[k][w]) fprintf(f, " %5s%d.w%-2d", "bz:i", k, w);
    fprintf(f, "\n");
    for (long i = lo + 1; i < head; i++) {
        const FrSnap* a = &s_fr[(i - 1) % FR_N]; const FrSnap* b = &s_fr[i % FR_N];
        long long dp = b->pres - a->pres, tw = 0;
        for (int w = 0; w < 17; w++) tw += b->tw_us[w] - a->tw_us[w];
        fprintf(f, "%9.3f %6.1f %4lld %5u %7.0f %5lld %4lld %6.1f %7.1f", b->t, (b->t - a->t) * 1000.0, dp, b->flips - a->flips,
                dp ? (double)(b->draws - a->draws) / dp : 0.0, b->reads - a->reads, b->ef_n - a->ef_n, (b->ef_us - a->ef_us) / 1000.0, tw / 1000.0);
        for (int w = 0; w < 17; w++) if (twc[w]) fprintf(f, " %9.1f", (b->tw_us[w] - a->tw_us[w]) / 1000.0);
        for (int k = 0; k < 2; k++) for (int w = 0; w < 16; w++) if (bc[k][w]) fprintf(f, " %10.1f", (b->busy[k][w] - a->busy[k][w]) / 1000.0);
        fprintf(f, "\n");
    }
    fclose(f);
    fprintf(stderr, "[FRAMEREC] wrote %s (%ld slices)\n", path, head - lo - 1);
}
static void fr_start(void)
{
    static volatile long once = 0;
    if (once || InterlockedExchange(&once, 1)) return;
    const char* e = getenv("FRAMEREC");
    if (!e || *e != '1') return;
    g_ps3_fr_dump = fr_dump;
    HANDLE h = CreateThread(NULL, 0, fr_thread, NULL, 0, NULL);
    if (h) CloseHandle(h);
    fprintf(stderr, "[FRAMEREC] v113 on: 50 ms slices, F8 -> framerec_N.txt (last ~6 s) + main-thread profile of the last 3 s\n");
}
static SPU_PM_TLS unsigned t_tw_k;   /* v132: rescans in the open wait */
/* QC_TL 'W' events: the waited object is r31 in inFamous 2's second CTaskMgr loop (func_0037062C, lr 0x0037xxxx at
 * begin) and r28 in the first (func_00371054); the workload is the one whose live arg equals one of its first 16 words. */
static SPU_PM_TLS LONGLONG t_qc_tw0;
static SPU_PM_TLS u32 t_qc_lr, t_qc_caller, t_qc_obj, t_qc_side; static SPU_PM_TLS int t_qc_wid;
static void qc_tw_begin(void)
{
    t_qc_tw0 = 0;
    if (!g_qc_rec) return;
    ppu_context* c = g_active_ctx;
    t_qc_tw0 = qc_now(); t_qc_wid = 16; t_qc_obj = 0; t_qc_caller = 0; t_qc_lr = 0; t_qc_side = 0;
    if (!c) return;
    t_qc_lr = (u32)c->lr;
    const u32 sp = (u32)c->gpr[1];
    if (sp >= 0x10000u && sp < 0xF0000000u) { const u32 osp = vm_read32(sp + 4); if (osp >= 0x10000u && osp < 0xF0000000u) t_qc_caller = vm_read32(osp + 0x14); }
    const int second = (t_qc_lr & 0xFFFFF000u) == 0x00370000u;
    const u32 o = (u32)c->gpr[second ? 31 : 28];
    t_qc_side = (u32)c->gpr[29] & 0xFF;
    t_qc_obj = o;
    if (o < 0x10000u || o >= 0xF0000000u) return;
    for (u32 off = 0; off < 0x40; off += 4) {
        const u32 v = vm_read32(o + off);
        if (v < 0x10000u) continue;
        for (int w = 0; w < 16; w++) if (s_workloads[w].in_use) {
            const u64 a = vm_read64(s_workloads[w].spurs_ea + SPURS_WKL_INFO1 + (u32)w * SPURS_WKL_INFO_SZ + 8);
            if ((u32)a == v || (u32)(a >> 32) == v) { t_qc_wid = w; return; }
        }
    }
}
static SPU_PM_TLS int t_tw_bw = -1;   /* v132 ABX 0x04: wid counted in g_tw_wid_n by this wait */
void ps3_taskwait_begin(void)
{
    fr_start();   /* v113 */
    qc_tw_begin();   /* QC_TL */
    t_tw_t0 = tw_now_us();
    g_spurs_tw_open_us = t_tw_t0;
    t_tw_marked = 0; t_tw_k = 0;
    t_tw_wid = spu_laneprof_on() ? tw_wid_of_wait() : -1;   /* v97f [TWCUR] */
    if (g_abx & 0x04) {   /* v132: mark the waited workload, wake its parked kernel at once */
        int w = t_tw_wid >= 0 ? t_tw_wid : tw_wid_of_wait();
        if (w < 0 || w > 16) w = 16;
        /* v133 TASKWAIT_W16_WID=<wid>: waits [TWCUR] cannot attribute (the hidden job poll in func_00287058; play132
         * LANEPROF: w2 BatchJobList is busy during them) are boosted as waits on that workload. */
        { static int s_w16 = -2; if (s_w16 == -2) { const char* e = getenv("TASKWAIT_W16_WID"); s_w16 = (e && *e) ? atoi(e) : -1; }
          if (w == 16 && s_w16 >= 0 && s_w16 < 16) w = s_w16; }
        t_tw_bw = w; InterlockedIncrement(&g_tw_wid_n[w]);
        ps3_spurs_kick();
    }
    if (spurs_tw_sweep_on()) { InterlockedIncrement(&g_spurs_ppu_tw); t_tw_marked |= 2; }
    if (spurs_taskwait_mode()) { InterlockedIncrement(&g_spurs_ppu_waiting); t_tw_marked |= 1; ps3_spurs_kick(); }
}
void ps3_taskwait_spin(void)
{
    s_tw_spins++;
    YieldProcessor();   /* the SMT sibling may be the lane we wait for */
    /* v132 ABX 0x08 "less spinning": after ~64 rescans (~20 us) pause 16x per rescan -- the main thread (HIGHEST,
     * P-core) re-scanned the job list ~1.8M times/s, taking issue slots and cache lines from the SPU threads it waits on */
    if ((g_abx & 0x08) && ++t_tw_k > 64) for (int i = 0; i < 15; i++) YieldProcessor();
}
void ps3_taskwait_end(void)
{
    if (t_qc_tw0) { qc_ev('W', (u32)t_qc_wid, t_qc_side, t_qc_tw0, qc_now(), t_qc_lr, t_qc_caller, t_qc_obj); t_qc_tw0 = 0; }   /* QC_TL */
    if (t_tw_bw >= 0) { InterlockedDecrement(&g_tw_wid_n[t_tw_bw]); t_tw_bw = -1; }   /* v132 */
    if (t_tw_marked & 1) InterlockedDecrement(&g_spurs_ppu_waiting);
    if (t_tw_marked & 2) InterlockedDecrement(&g_spurs_ppu_tw);
    g_spurs_tw_open_us = 0;
    t_tw_marked = 0;
    long long now = tw_now_us(), d = now - t_tw_t0;
    if (t_tw_wid >= 0 && t_tw_wid <= 16) { g_fr_tw_us[t_tw_wid] += d; g_fr_tw_n[t_tw_wid]++; }   /* v113 */
    else { g_fr_tw_us[16] += d; g_fr_tw_n[16]++; }
    s_tw_n++; s_tw_us += d; if (d > s_tw_max) s_tw_max = d; if (d < 5) s_tw_fast++;
    if (t_tw_wid >= 0 && t_tw_wid <= 16) { const int w = t_tw_wid;   /* v97f [TWCUR] */
        s_tww_n[w]++; s_tww_us[w] += d; if (d >= 100) s_tww_long[w]++; if (d > s_tww_max[w]) s_tww_max[w] = d; }
    t_tw_wid = -1;
    if (!s_tw_win0) s_tw_win0 = now;
    else if (now - s_tw_win0 >= 10000000LL) {
        double w = (double)(now - s_tw_win0);
        fprintf(stderr, "[TASKWAIT] 10s: %lld waits, %.0f ms waiting (%.1f%% of wall), mean %.0f us, max %lld us, %lld done < 5 us, %lld spins (ab phase %d abx 0x%02X)\n",
                s_tw_n, s_tw_us / 1000.0, 100.0 * s_tw_us / w, s_tw_n ? (double)s_tw_us / s_tw_n : 0.0, s_tw_max, s_tw_fast, s_tw_spins, g_ab_mask, g_abx);
        {   /* v97f [TWCUR]: the same 10 s window split by waited workload (w16 = no match) */
            char b[1024]; int p = snprintf(b, sizeof b, "[TWCUR] 10s waits by workload:");
            for (int w = 0; w <= 16; w++) {
                if (s_tww_n[w] && p < (int)sizeof b - 80)
                    p += snprintf(b + p, sizeof b - p, " | w%d %lld waits %.0f ms (%lld >= 100us, max %.1f ms)", w, s_tww_n[w],
                                  s_tww_us[w] / 1000.0, s_tww_long[w], s_tww_max[w] / 1000.0);
                s_tww_n[w] = s_tww_us[w] = s_tww_long[w] = s_tww_max[w] = 0;
            }
            if (p > 40) fprintf(stderr, "%s (ab %d)\n", b, g_ab_mask);
        }
        s_tw_n = s_tw_us = s_tw_max = s_tw_fast = s_tw_spins = 0; s_tw_win0 = now;
    }
}
void ps3_spurs_kick(void)
{
    if (s_kick_parked <= 0) return;
    long n = s_kick_n; if (n > SPURS_KICK_MAX) n = SPURS_KICK_MAX;
    long woke = 0;
    for (long i = 0; i < n; i++)
        if (s_kick_waiting[i] && InterlockedExchange(&s_kick_waiting[i], 0)) {
#ifdef _WIN32
            SetEvent(s_kick_ev[i]);
#endif
            woke++;
        }
    if (woke) { InterlockedExchangeAdd(&s_kick_woke, woke); InterlockedIncrement(&s_kick_calls); }
}
/* v140 SPURS_INST1_HOT (default 1; 0 = old): instance 1 = inFamous' AUDIO mixer (w3 BrBDblBufWrkld, nSpus=1 --
 * a dedicated SPU on the console). play139: the audio holes track w3's load exactly (late holes 0-90 per 5 s at
 * <= 7 s busy per 10 s, 250-450 at 9.2-9.4 s) and the user hears them with Cole's lightning (heavy mixing), while
 * the w3 thread also gave its CPU away: SwitchToThread after EVERY busy pass (~11k/s) at ABOVE_NORMAL among ~20
 * busy P-core threads, plus timed naps when idle (16% of its samples). Hot: HIGHEST priority, never yields after
 * a busy pass, spins (PAUSE) up to 2 ms after the last work before it naps -> one logical CPU acts as the SPU. */
static int spurs_inst1_hot(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("SPURS_INST1_HOT"); v = (e && *e == '0') ? 0 : 1;
                 fprintf(stderr, "[spurs-kern] v140 audio kernel (inst1/w3): %s (SPURS_INST1_HOT)\n", v ? "HOT (HIGHEST, no yield, 2 ms spin)" : "old"); }
    return v;
}
static DWORD WINAPI spurs_kernel_thread(LPVOID p)
{
#ifdef _WIN32
    { ULONG g = 256 * 1024; SetThreadStackGuarantee(&g); }  /* let SO reach the reporter */
#endif
    { extern void ps3_prof_register_spu_thread(void); ps3_prof_register_spu_thread(); }   /* v63 PS3_PROF_MAIN */
    struct SpursKArg* ka = (struct SpursKArg*)p;
    struct SpursInst* si = ka->si;
    const u32 k_part = ka->part, k_nparts = ka->nparts;
#ifdef _WIN32
    /* Instance 0 / part 0 owns inFamous' main CTaskMgr workload (wid 0).
     * Match its lane helpers' policy without forcing the other kernel threads
     * off the E-cores. */
    if ((int)(si - s_inst) == 0 && k_part == 0) {
        const char* e = getenv("PS3_SPURS_PCORE");
        if (e && *e && *e != '0') {
            extern void ps3_thread_prefer_pcores(const char*, int);
            ps3_thread_prefer_pcores("SPURS w0 kernel", THREAD_PRIORITY_ABOVE_NORMAL);
        }
    }
    /* v112 SPURS_INST1_PCORE (default 1; 0 = off): instance 1 (inFamous: nSpus=1, its only workload is w3
     * BrBDblBufWrkld, the renderer's double-buffer job) ran its single kernel thread unpinned at NORMAL
     * priority -> E-cores. play110: that thread is 74-78% busy in calm 10 s windows and 87-95% in the windows
     * with the world-movement hitches (Cole keeps simulating; the drawn position freezes, then snaps). */
    if ((int)(si - s_inst) == 1 && k_part == 0) {
        const char* e = getenv("SPURS_INST1_PCORE");
        if (!e || *e != '0') {
            extern void ps3_thread_prefer_pcores(const char*, int);
            ps3_thread_prefer_pcores("SPURS inst1 kernel (w3 BrBDblBufWrkld)", spurs_inst1_hot() ? THREAD_PRIORITY_HIGHEST : THREAD_PRIORITY_ABOVE_NORMAL);
        }
    }
    if ((int)(si - s_inst) == 1 && k_part == 0) { extern void ps3_prof_register_inst1_thread(void); ps3_prof_register_inst1_thread(); }   /* v138 */
    if ((int)(si - s_inst) == 1 && k_part == 0 && spurs_inst1_hot()) { extern void ps3_audio_core_claim(void); ps3_audio_core_claim(); }   /* v141: own physical core */
#endif
    int kick_slot = -1;         /* v82 SPURS_KICK slot */
#ifdef _WIN32
    {   /* v83: always register (TASKWAIT_BOOST uses the same wake path) */
        long slot = InterlockedIncrement(&s_kick_n) - 1;
        if (slot < SPURS_KICK_MAX) {
            s_kick_ev[slot] = CreateEventA(NULL, FALSE, FALSE, NULL);
            if (s_kick_ev[slot]) kick_slot = (int)slot;
        }
    }
#endif
    u32 k_pass_no = 0;          /* per-thread pass counter (idle cadence) */
    u32 k_owned = 0, k_owned_new = 0;   /* v132: wids this kernel thread dispatched last pass (ABX 0x04) */
    int k_busy_pass = 0;        /* last pass ran a module that found work */
    static volatile long s_pm_off = -1;
    if (s_pm_off < 0) s_pm_off = getenv("PS3_NO_SPURS_PM") ? 1 : 0;

    fprintf(stderr, "[spurs-kern] \"%s\" poll thread live: ea=0x%08X pm=%s\n",
            si->prefix, si->ea, s_pm_off ? "DISABLED (PS3_NO_SPURS_PM)" : "enabled");
    fflush(stderr);

    /* Instance change detector (SPURS_KERN_WATCH=1).
     *
     * We assumed the title kicks a workload by poking wklReadyCount/wklSignal
     * with inlined atomics -- but this thread watches exactly those bytes every
     * 1 ms and has never once seen them nonzero, across whole runs. Rather than
     * guess again, shadow the head of the instance and report EVERY byte the
     * title changes. Whatever the real kick is, it has to land in here. */
    static const u32 WATCH_LEN = 0xC0;
    unsigned char shadow[0xC0];
    int shadow_primed = 0;
    int watch = getenv("SPURS_KERN_WATCH") ? 1 : 0;
    int changes_logged = 0;
    /* v127 PCORE_UNPIN_PHASE (default 12): while g_ab_mask == that phase, the instance-1 kernel thread (w3, ~95% busy,
     * pinned by v112 SPURS_INST1_PCORE during the freeze hunt) runs on all CPUs at NORMAL priority instead of the
     * P-cores -> in-run A/B of freeing a P-core for the main thread. Checked once per kernel pass (int compare). */
    int k_inst1_pin = 0, k_inst1_state = 1, k_unpin_ph = 12; (void)k_inst1_pin; (void)k_inst1_state; (void)k_unpin_ph;
#ifdef _WIN32
    { const char* e1 = getenv("SPURS_INST1_PCORE"); const char* e2 = getenv("PCORE_UNPIN_PHASE");
      k_inst1_pin = ((int)(si - s_inst) == 1 && k_part == 0 && !(e1 && *e1 == '0'));
      if (e2 && *e2) k_unpin_ph = atoi(e2); }
#endif

    int k_w0_pin = -1;   /* v132 ABX 0x01: -1 n/a, else current pin state of the w0 kernel (inst 0 part 0) */
#ifdef _WIN32
    { const char* e = getenv("PS3_SPURS_PCORE"); if ((int)(si - s_inst) == 0 && k_part == 0 && e && *e && *e != '0') k_w0_pin = 1; }
#endif
    for (;;) {
#ifdef _WIN32
        if (k_w0_pin >= 0) { const int want = !(g_abx & 0x40);
            if (want != k_w0_pin) { extern void ps3_thread_pcore_set(const char*, int, int);
                k_w0_pin = want; ps3_thread_pcore_set("SPURS w0 kernel", want, THREAD_PRIORITY_ABOVE_NORMAL); } }
        if (k_inst1_pin) {   /* v127 */
            extern volatile int g_ab_mask;
            const int want = (g_ab_mask != k_unpin_ph);
            if (want != k_inst1_state) { k_inst1_state = want;
                extern void ps3_thread_pcore_set(const char*, int, int);
                ps3_thread_pcore_set("SPURS inst1 kernel (w3)", want, spurs_inst1_hot() ? THREAD_PRIORITY_HIGHEST : THREAD_PRIORITY_ABOVE_NORMAL); }
        }
#endif
        /* Yield instead of sleeping after a pass that did work: a CTaskMgr task
         * chain needs one pass per step, and a 1 ms sleep per step stacked up
         * on the main thread's per-frame wait. Idle passes still sleep. */
        /* SPURS_SPIN_US (2026-10-04, default 4000): keep polling with yields for this
         * long after the last pass that found work before falling back to Sleep(1).
         * Profile: kernel threads were 35-40% asleep while the main thread waited
         * on them -- each dependency hop of a CTaskMgr chain (often across
         * workloads/threads) paid a >= 1 ms Sleep. =0 -> old behaviour. */
        {
            static LONGLONG s_freq = 0, s_spin_ticks = -1;
            if (!s_freq) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); s_freq = f.QuadPart; }
            if (s_spin_ticks < 0) { const char* e = getenv("SPURS_SPIN_US"); long us = e ? atol(e) : 500;   /* v28: 4000 -> 30000; v67: 2000; v68: 500 (play66/67 A/B) */
                                    s_spin_ticks = (LONGLONG)us * s_freq / 1000000; }
            static SPU_PM_TLS LONGLONG t_last_busy = 0;
            LARGE_INTEGER now; QueryPerformanceCounter(&now);
            /* v60b telemetry: share of wall time this kernel thread spent in passes that
             * found work vs passes that found nothing (spin). PS3_TELEM=0 off. */
            static SPU_PM_TLS LONGLONG t_pass0 = 0, t_win0 = 0, t_busy = 0, t_idle = 0;
            { static int s_tm = -1; if (s_tm < 0) { const char* e = getenv("PS3_TELEM"); s_tm = (e && *e == '0') ? 0 : 1; }
              if (s_tm) {
                if (t_pass0) { if (k_busy_pass) t_busy += now.QuadPart - t_pass0; else t_idle += now.QuadPart - t_pass0; }
                if (!t_win0) t_win0 = now.QuadPart;
                if (now.QuadPart - t_win0 >= s_freq * 10) {
                    double w = (double)(now.QuadPart - t_win0);
                    fprintf(stderr, "[SPURS-TELEM] \"%s\" part %u hosttid %lu: work passes %.0f%%, empty passes %.0f%%, asleep %.0f%%\n",
                            si->prefix, k_part, (unsigned long)GetCurrentThreadId(), 100.0 * t_busy / w, 100.0 * t_idle / w,
                            100.0 * (w - t_busy - t_idle) / w);
                    t_win0 = now.QuadPart; t_busy = t_idle = 0;
                }
              }
            }
            if (kick_slot == 0) {   /* v82: one [SPURS-KICK] line per 10 s */
                static LONGLONG t_kr = 0;
                if (!t_kr) t_kr = now.QuadPart;
                else if (now.QuadPart - t_kr >= s_freq * 10) {
                    long w = InterlockedExchange(&s_kick_woke, 0), c = InterlockedExchange(&s_kick_calls, 0);
                    fprintf(stderr, "[SPURS-KICK] 10s: %ld polls woke %ld parked kernel threads (%ld threads registered, ab phase %d)\n",
                            c, w, s_kick_n, g_ab_mask);
                    t_kr = now.QuadPart;
                }
            }
            int tw_spin = 0;
            if (!k_busy_pass && g_spurs_ppu_waiting > 0) {
                int m = spurs_taskwait_mode();
                tw_spin = m == 2 || (m == 1 && (int)(si - s_inst) == 0 && k_part == 0);
            }
            if (!k_busy_pass && !tw_spin && (g_abx & 0x8000)) {   /* NB16 NB_HOT_WIDS=<wid,..> (default 2): owner kernels never nap */
                static u32 s_hot = 0xFFFFFFFFu;
                if (s_hot == 0xFFFFFFFFu) { u32 m = 0; const char* e = getenv("NB_HOT_WIDS"); if (!e) e = "2";
                    while (*e) { while (*e == ',' || *e == ' ') e++; if (!*e) break; int w = atoi(e); if (w >= 0 && w < 16) m |= 1u << w; while (*e && *e != ',') e++; }
                    s_hot = m; }
                if (k_owned & s_hot) tw_spin = 1;
            }
            if (!k_busy_pass && !tw_spin && (g_abx & 0x04))   /* v132: no nap while the PPU waits on one of OUR workloads */
                for (u32 w = 0; w < 16; w++) if (((k_owned >> w) & 1) && g_tw_wid_n[w] > 0) { tw_spin = 1; break; }
            const int k_hot = ((int)(si - s_inst) == 1 && k_part == 0 && spurs_inst1_hot());   /* v140 */
            if (k_hot && (k_busy_pass || now.QuadPart - t_last_busy < s_freq / 500)) {
                if (k_busy_pass) t_last_busy = now.QuadPart;
                else for (int z = 0; z < 32; z++) YieldProcessor();   /* idle: short PAUSE spin, keep the CPU */
            }
            else if (k_busy_pass || tw_spin) { t_last_busy = now.QuadPart; hp4b_spurs_spin(); }   /* HP4B: was SwitchToThread() */
            /* v68: idle policy from the play66/67 A/B: spin (yield) only SPURS_SPIN_US after the last
             * busy pass (default 500 us), then wait SPURS_IDLE_WAIT_US (default 250 us, high-res
             * timer) per empty pass instead of Sleep(1). PS3_AB=1 cycles 5 (spin, wait) pairs. */
            else {
                static const int ab_spin[5] = { -1, -1, -1, 2000, -1 }, ab_wait[5] = { -1, -1, -1, 50, -1 };   /* v93: phase 3 = boost owner + 2 ms spin / 50 us naps */   /* v83: PS3_AB phases select TASKWAIT_BOOST modes, idle policy fixed */
                static LONGLONG s_wait_us = -1;
                if (s_wait_us < 0) { const char* e = getenv("SPURS_IDLE_WAIT_US"); s_wait_us = e ? atol(e) : 250; }
                const int ph = g_ab_mask; /* phase index 0..4 under PS3_AB, else 0 */
                LONGLONG spin_t = (ph > 0 && ph < 5 && ab_spin[ph] >= 0) ? (LONGLONG)ab_spin[ph] * s_freq / 1000000 : s_spin_ticks;
                if ((g_abx & 0x08) && spin_t > s_freq / 10000) spin_t = s_freq / 10000;   /* v132 ABX 0x08: kernel spin window 500 -> 100 us */
                LONGLONG wait_us = (ph > 0 && ph < 5 && ab_wait[ph] >= 0) ? ab_wait[ph] : s_wait_us;
                if (now.QuadPart - t_last_busy < spin_t) hp4b_spurs_spin();   /* HP4B: was SwitchToThread() */
#ifdef _WIN32
                else if (kick_slot >= 0 && wait_us > 0) {
                    /* v82: timed wait that a guest completion poll can cut short */
                    static SPU_PM_TLS HANDLE t_tm2 = NULL; static SPU_PM_TLS int t_tm2_tried = 0;
                    if (!t_tm2_tried) { t_tm2_tried = 1; t_tm2 = CreateWaitableTimerExW(NULL, NULL, 0x00000002, 0x1F0003); }
                    InterlockedExchange(&s_kick_waiting[kick_slot], 1);
                    InterlockedIncrement(&s_kick_parked);
                    const LONGLONG qc_n0 = g_qc_rec ? qc_now() : 0;   /* QC_TL */
                    DWORD r;
                    if (t_tm2) {
                        LARGE_INTEGER due; due.QuadPart = -(wait_us * 10);
                        SetWaitableTimer(t_tm2, &due, 0, NULL, NULL, FALSE);
                        HANDLE hs[2] = { s_kick_ev[kick_slot], t_tm2 };
                        r = WaitForMultipleObjects(2, hs, FALSE, INFINITE);
                    } else {
                        r = WaitForSingleObject(s_kick_ev[kick_slot], (DWORD)((wait_us + 999) / 1000));
                    }
                    InterlockedDecrement(&s_kick_parked);
                    if (qc_n0) qc_ev('N', 0, k_part, qc_n0, qc_now(), r == WAIT_OBJECT_0 ? 1u : 0u, (u32)wait_us, k_owned);   /* QC_TL */
                    /* kicked (WAIT_OBJECT_0): run ONE pass now; only a pass that finds work restarts the
                     * spin window, so a long PPU poll does not turn the parked threads into spinners */
                    if (r != WAIT_OBJECT_0 && InterlockedExchange(&s_kick_waiting[kick_slot], 0) == 0) {
                        /* a kicker claimed the slot after the timer fired: drain its SetEvent */
                        WaitForSingleObject(s_kick_ev[kick_slot], 0);
                    }
                }
#endif
                else if (wait_us >= 1000) Sleep((DWORD)(wait_us / 1000));
                else {
#ifdef _WIN32
                    static SPU_PM_TLS HANDLE t_tm = NULL; static SPU_PM_TLS int t_tm_tried = 0;
                    if (!t_tm_tried) { t_tm_tried = 1; t_tm = CreateWaitableTimerExW(NULL, NULL, 0x00000002, 0x1F0003); }
                    if (t_tm && wait_us > 0) { const LONGLONG qc_n1 = g_qc_rec ? qc_now() : 0;   /* QC_TL */
                        LARGE_INTEGER due; due.QuadPart = -(wait_us * 10); SetWaitableTimer(t_tm, &due, 0, NULL, NULL, FALSE); WaitForSingleObject(t_tm, INFINITE);
                        if (qc_n1) qc_ev('N', 0, k_part, qc_n1, qc_now(), 2u, (u32)wait_us, k_owned); }
                    else if (wait_us > 0) Sleep(1); else SwitchToThread();
#else
                    if (wait_us > 0) usleep((useconds_t)wait_us); else sched_yield();
#endif
                }
            }
            { LARGE_INTEGER tp; QueryPerformanceCounter(&tp); t_pass0 = tp.QuadPart; }
        }
        k_busy_pass = 0;
        k_owned = k_owned_new; k_owned_new = 0;   /* v132 */
        k_pass_no++;
        qc_init();   /* QC_TL (one load after the first call) */
        u32 ea = si->ea;
        if (!ea || s_pm_off) continue;

        if (watch && k_part == 0) {
            const unsigned char* live = (const unsigned char*)vm_base + ea;
            if (!shadow_primed) { memcpy(shadow, live, WATCH_LEN); shadow_primed = 1; }
            else if (memcmp(shadow, live, WATCH_LEN) != 0) {
                for (u32 o = 0; o < WATCH_LEN; o++) {
                    if (shadow[o] == live[o]) continue;
                    if (changes_logged < 200) {
                        changes_logged++;
                        fprintf(stderr, "[spurs-kern] \"%s\" INSTANCE +0x%02X: %02X -> %02X%s\n",
                                si->prefix, o, shadow[o], live[o],
                                o < 16                    ? "  (wklReadyCount1)" :
                                o >= 0x70 && o < 0x76     ? "  (wklSignal1)"     :
                                o >= 0x80 && o < 0x90     ? "  (wklState1)"      :
                                o >= 0xB0 && o < 0xB4     ? "  (wklEnabled)"     : "");
                    }
                }
                memcpy(shadow, live, WATCH_LEN);
                fflush(stderr);
            }
        }

        u32 enabled = vm_read32(ea + SPURS_WKL_ENABLED);

        /* Kick visibility (can't-miss): log ANY nonzero readyCount/signal state
         * even for wids the dispatch filter below would skip — the game pokes
         * these bytes with inlined atomics and this is our only tap. */
        {
            static int _seen[MAX_SPURS_INST][16];
            int slot = (int)(si - s_inst);
            u32 sig = vm_read32(ea + SPURS_WKL_SIGNAL1) >> 16;
            for (u32 w = 0; w < 16; w++) {
                u8 rc = *(vm_base + ea + SPURS_WKL_READY1 + w);
                if (k_part == 0 && (rc || (sig & (0x8000u >> w))) && _seen[slot][w] < 4) {
                    _seen[slot][w]++;
                    fprintf(stderr, "[spurs-kern] \"%s\" POKE wid=%u ready=%u sig=%u enabled=%d state=%u\n",
                            si->prefix, w, rc, (sig >> (15 - w)) & 1,
                            (enabled >> (31 - w)) & 1,
                            *(vm_base + ea + SPURS_WKL_STATE1 + w));
                }
            }
        }
        if (!enabled) continue;

        for (u32 wid = 0; wid < 16; wid++) {
            if (!(enabled & (0x80000000u >> wid))) continue;
            if (*(vm_base + ea + SPURS_WKL_STATE1 + wid) != 2) continue;
            if (!s_workloads[wid].in_use || s_workloads[wid].spurs_ea != ea) continue;
            if (spurs_wkl_part(wid, k_nparts, si->solo) != k_part) continue;   /* SPURS_WKL_THREADS partition (v29: + SOLO/MAP) */
            k_owned_new |= 1u << wid;   /* v132 */

            /* A SPURS policy module is a PERSISTENT SPU program. The real kernel
             * schedules an ENABLED, runnable workload onto an SPU and the module
             * then polls its OWN job queue in main memory; being enabled is the
             * trigger, not a per-job kick. That is why this title never writes
             * wklReadyCount, never sets a signal bit, and never calls
             * cellSpursReadyCountStore -- on hardware it does not have to. Gating
             * dispatch on a kick meant the module never ran at all, so nothing
             * ever called cellSpursEventFlagSet and the title's loading thread
             * blocked forever.
             *
             * Run one scheduling quantum per enabled workload per pass: the
             * module does its work, exits to the kernel (LS 0x9C0) when it has
             * none, and we re-enter it on the next pass -- which is exactly what
             * the real kernel's dispatch loop does.
             *
             * readyCount/wklSignal are still honoured when a title DOES use them:
             * consume one unit so a kick-driven title paces the same as before. */
            volatile u8* rdy = vm_base + ea + SPURS_WKL_READY1 + wid;
            u32 sig = vm_read32(ea + SPURS_WKL_SIGNAL1) >> 16;    /* be u16 @0x70 */
            int kicked = (*rdy != 0) || ((sig & (0x8000u >> wid)) != 0);
            if (*rdy) (*rdy)--;
            if (sig & (0x8000u >> wid))
                vm_write32(ea + SPURS_WKL_SIGNAL1,
                           (vm_read32(ea + SPURS_WKL_SIGNAL1) & ~((0x8000u >> wid) << 16)));

            WklPm* r = spurs_resolve_pm(wid);
            if (!r) continue;

            /* Idle backoff: running EVERY enabled workload's module EVERY 1ms
             * pass (x N instance threads) burned ~5 host cores on modules that
             * immediately exit-to-kernel with no work, starving the actual
             * decode/render threads (LBP movie at ~1fps while 500% CPU).
             * A module that keeps finding nothing gets re-run every 2nd, 4th,
             * ... up to 16th pass; an explicit kick (readyCount/signal) resets
             * it to every pass, so kick-driven latency is unchanged. */
            {
                static u8 s_idle[CELL_SPURS_MAX_WORKLOAD];      /* idle streak (log2 cadence) */
                if (kicked) s_idle[wid] = 0;
                /* SPURS_IDLE_SHIFT (2026-10-04, default 1 = an idle workload is still
                 * checked every 2nd pass; was 4 = every 16th). New per-frame CTaskMgr work
                 * on an idle workload waited up to 16 passes to be noticed -- latency the
                 * main thread paid every frame. With SPURS_WKL_THREADS each thread has
                 * few workloads, so the extra idle PM runs are cheap. */
                static int s_ish = -1;
                if (s_ish < 0) { const char* e = getenv("SPURS_IDLE_SHIFT"); s_ish = e ? atoi(e) : 1; if (s_ish < 0) s_ish = 0; if (s_ish > 4) s_ish = 4; }
                u32 cad = 1u << (s_idle[wid] > (u8)s_ish ? (u32)s_ish : s_idle[wid]);
                if (!kicked && (k_pass_no & (cad - 1)) != 0) continue;
                extern SPU_PM_TLS volatile unsigned g_spurs_pm_polls;
                u32 polls_before = g_spurs_pm_polls;   /* heuristic only */
                (void)polls_before;

            {   static int _n = 0;
                if (_n < 8) { _n++;
                    fprintf(stderr, "[spurs-kern] \"%s\" dispatch wid=%u (enabled, state=2) "
                                    "image=%d ready=%u\n", si->prefix, wid, r->image_id, *rdy);
                    fflush(stderr); } }

            /* Live workload arg from the real wklInfo (the game may update it). */
            u64 arg = vm_read64(ea + SPURS_WKL_INFO1 + wid * SPURS_WKL_INFO_SZ + 8);
            {   /* v97b [STALL]: queue line = workload data EA (hi word when the data is a 64-bit pair) */
                u32 stl_ea = ((u32)(arg >> 32) ? (u32)(arg >> 32) : (u32)arg) & ~127u;
                if (stl_ea >= 0x10000u && stl_ea < 0xF0000000u) spurs_stall_pre((int)(si - s_inst), wid, stl_ea);
            }
            /* Dispatch once per VIRTUAL SPU up to the workload's maxContention:
             * the WWS job manager keys its per-SPU ticket lane off the kernel
             * context's spuNum, and its command lists carry cross-lane BARRIER
             * commands -- with only spu 0 ever dispatched, lane 1 (pre-armed by
             * the PPU for a 2-SPU workload) never advanced and every barrier
             * deadlocked: LBP's post-intro loading froze with 8 ready jobs and
             * the lanes stuck at {1,0,...} against ticket 10+. Sequential
             * per-lane rounds converge where parallel SPUs would. */
            u8 maxcont = *(vm_base + ea + SPURS_WKL_MAXCONT + wid);
            if (maxcont < 1) maxcont = 1;
            if (maxcont > 6) maxcont = 6;
            /* Do NOT exceed the workload's own contention: per-SPU rows in
             * the WWS sync struct live at +0x40+16*spuNum, and dispatching
             * spuNum >= nSpus made virtual SPU 3 write its bookkeeping row
             * OVER the game's ticket row (row 3) -- observed as ticket values
             * jumping to garbage (141028, 9960...) during the savedata load.
             * The earlier "4 lanes drains the queue" result was partly that
             * scribble. Correct progress comes from PERSISTENT PM contexts
             * (spurs_policy.c), not extra lanes. */
            /* SPURS_FORCE_SPUS=<n>: dispatch every workload for n virtual SPUs
             * regardless of maxContention (A/B: LBP publishes its loading
             * tickets on sync row 3, and the PM's row index = spuNum).
             * 2026-07-23 FINDING: with the completion HLE (LBP_HLE_JOBDONE) in
             * place, forcing 2+ lanes is now COUNTERPRODUCTIVE -- two host
             * threads race on the shared jobIndex atomic in WwsJob_AllocateJob
             * (GETLLAR 0xD0/PUTLLC 0xB4 on the joblist header), making the boot
             * nondeterministic. Single-lane (leave SPURS_FORCE_SPUS unset) is
             * both correct (RPCS3 oracle: jobmanager runs one SPU at a time) and
             * far more stable -- it reaches the furthest boot yet (LBP renders to
             * finish#922: glyphthread+network+camera up). PREFER single-lane. */
            { static int s_fs = -2;
              if (s_fs == -2) { const char* e = getenv("SPURS_FORCE_SPUS");
                s_fs = e ? atoi(e) : -1; }
              if (s_fs > 0) maxcont = (u8)(s_fs > 6 ? 6 : s_fs); }
            *(vm_base + ea + SPURS_WKL_CURCONT + wid) = maxcont;
            LARGE_INTEGER _pmt0 = {0}; if (spurs_stats_on()) QueryPerformanceCounter(&_pmt0);
            unsigned long long _spin0 = spurs_stats_on() ? g_spu_spin_polls : 0;
            const LONGLONG qc_k0 = g_qc_rec ? qc_now() : 0;   /* QC_TL */
            spurs_wkl_run_lock(wid);   /* v97f */
            if (!spurs_wkl_may_run(ea, wid, arg)) {   /* v102: shut down / removed since the check above */
                spurs_wkl_run_unlock(wid);
                *(vm_base + ea + SPURS_WKL_CURCONT + wid) = 0;
                continue;
            }
            {
                const uint8_t* pm = (const uint8_t*)vm_base + (uint32_t)(uintptr_t)s_workloads[wid].pm;
                uint32_t sz = s_workloads[wid].sizePm;
                if (maxcont <= 1) {
                    /* SPURS_SEQ_LANES (2026-10-03, default on): a real kernel
                     * runs a maxContention-1 workload on WHICHEVER of the
                     * instance's nSpus SPUs is free, so over time every SPU
                     * visits it. The WWS job manager keeps one progress lane
                     * per SPU (sync +0x40+row*16, lanes +2..), and the PPU's
                     * "batch done" check takes the MIN over the lanes of all
                     * nSpus SPUs. Always running as SPU 0 left SPU 1's lane
                     * behind forever: inFamous (nSpus=2) sat on its loading
                     * screen in func_0041DC00 / func_0041CF98 waiting for
                     * BatchJobList (wid 2). Visit each of the instance's SPUs
                     * in turn, one at a time (no concurrency -- see
                     * SPURS_FORCE_SPUS above for why parallel lanes race).
                     * SPURS_SEQ_LANES=0 restores SPU-0-only dispatch. */
                    static int s_seq = -1;
                    if (s_seq < 0) { const char* e = getenv("SPURS_SEQ_LANES"); s_seq = (e && *e == '0') ? 0 : 1; }
                    u32 nl = s_seq ? si->nspus : 1u;
                    if (nl < 1) nl = 1;
                    if (nl > 6) nl = 6;
                    const int parallel_sweep = spurs_parallel_sweep_allowed(wid, nl, maxcont);
                    /* v29: g_spurs_pm_polls is reset at the start of EVERY lane's run, so
                     * after this loop it only described the LAST lane: a workload whose work
                     * was done by lanes 0..n-2 counted as idle (wrong idle backoff, wrong
                     * k_busy_pass, [pm-time] busy ~0 for w0). Sum over the lanes. */
                    unsigned polls_sum = 0;
                    /* v30 SPURS_IDLE_ROTATE (default on; =0 off): once a workload has come
                     * up empty on two full sweeps in a row (and is not kicked), visit ONE
                     * lane per pass, rotating 0..nl-1, instead of all nl. Every lane is
                     * still visited in the same cyclic order; the first lane that finds
                     * work resets the streak and full sweeps resume on the next pass.
                     * [pm-time] v29: an empty 5-lane sweep costs ~40 us, and idle
                     * LibMP3 (w4) alone burned 30-38% of the kernel thread it shares
                     * with w9; w0's idle sweeps were ~25% of its own thread. */
                    static int s_irot = -1;
                    if (s_irot < 0) { const char* e = getenv("SPURS_IDLE_ROTATE"); s_irot = (e && *e == '0') ? 0 : 1; }
                    static u8 s_rot[CELL_SPURS_MAX_WORKLOAD];
                    /* v132 ABX 0x04: full sweeps only for the workload the PPU actually waits on (or all when unattributed) */
                    const int tw_full = (g_abx & 0x04) ? (g_tw_wid_n[wid] > 0 || g_tw_wid_n[16] > 0) : (g_spurs_ppu_tw > 0);
                    if (s_irot && nl > 1 && !kicked && s_idle[wid] >= 2 && !tw_full) {   /* v92: full sweeps while the PPU waits */
                        u32 sn = (u32)(s_rot[wid]++) % nl;
                        if (parallel_sweep && spurs_par_kick_on()) {
                            polls_sum = spurs_run_lane_kickable(wid, nl, sn, r->fn, r->image_id, pm, sz, arg, ea);
                        } else {
                            spu_run_policy_module(r->fn, r->image_id, pm, sz, arg, wid, ea, sn);
                            polls_sum = g_spurs_pm_polls;
                        }
                    } else if (parallel_sweep) {
                        polls_sum = spurs_run_lanes_parallel(wid, nl, r->fn, r->image_id, pm, sz, arg, ea);
                    } else {
                        for (u32 sn = 0; sn < nl; sn++) {
                            spu_run_policy_module(r->fn, r->image_id, pm, sz, arg, wid, ea, sn);
                            polls_sum += g_spurs_pm_polls;
                        }
                    }
                    g_spurs_pm_polls = polls_sum;
                } else {
                    /* Run the workload's virtual SPUs CONCURRENTLY (see
                     * spurs_pm_worker): each lane advances in parallel so the
                     * cross-SPU barrier resolves. Spawn maxcont-1 workers for
                     * lanes 1..N-1 and run lane 0 on this thread, then join. */
                    HANDLE th[8]; struct spurs_pm_worker_arg wa[8];
                    unsigned nth = 0;
                    for (u32 sn = 1; sn < maxcont && nth < 7; sn++, nth++) {
                        wa[nth].fn = r->fn; wa[nth].image_id = r->image_id;
                        wa[nth].pm = pm; wa[nth].pm_size = sz;
                        wa[nth].arg = arg; wa[nth].wid = wid; wa[nth].ea = ea;
                        wa[nth].spu_num = sn;
                        th[nth] = CreateThread(NULL, 1u << 20, spurs_pm_worker, &wa[nth], 0, NULL);
                    }
                    spu_run_policy_module(r->fn, r->image_id, pm, sz, arg, wid, ea, 0);
                    if (nth) {
                        WaitForMultipleObjects(nth, th, TRUE, INFINITE);
                        for (unsigned k = 0; k < nth; k++) CloseHandle(th[k]);
                    }
                }
            }
            *(vm_base + ea + SPURS_WKL_CURCONT + wid) = 0;
            spurs_wkl_after_run(ea, wid, g_spurs_pm_polls);   /* v102: SHUTTING_DOWN -> REMOVABLE when the quantum ends */
            spurs_wkl_run_unlock(wid);   /* v97f */
            if (qc_k0) qc_ev('K', wid, k_part, qc_k0, qc_now(), g_spurs_pm_polls, (u32)arg, maxcont);   /* QC_TL */
            /* v28b SPURS_PMTIME (diagnostic opt-in for title runners; =0 off): host time per workload, reported
             * every 10 s by kernel thread 0 -- names the workload that serialises a
             * kernel thread (one thread at ~92% while the rest idle). */
            {
                if (spurs_stats_on()) {
                    static LONGLONG s_tsum[MAX_SPURS_INST][16], s_tmax[MAX_SPURS_INST][16];
                    static unsigned s_runs[MAX_SPURS_INST][16], s_busy[MAX_SPURS_INST][16];
                    static unsigned long long s_spin[MAX_SPURS_INST][16];   /* v30c */
                    static LONGLONG s_freq2 = 0, s_last_rep[MAX_SPURS_INST];   /* v29: per instance (shared one starved inst 0's report: its sums spanned ~12 windows) */
                    LARGE_INTEGER t1; QueryPerformanceCounter(&t1);
                    if (!s_freq2) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); s_freq2 = f.QuadPart; }
                    int sl = (int)(si - s_inst);
                    if (!s_last_rep[sl]) s_last_rep[sl] = t1.QuadPart;
                    LONGLONG d = t1.QuadPart - _pmt0.QuadPart;
                    s_tsum[sl][wid] += d; s_runs[sl][wid]++;
                    if (d > s_tmax[sl][wid]) s_tmax[sl][wid] = d;
                    if (g_spurs_pm_polls) s_busy[sl][wid]++;
                    if (sl < 2) { g_fr_runs[sl][wid]++; if (g_spurs_pm_polls) g_fr_busy_us[sl][wid] += d * 1000000 / s_freq2; }   /* v113 */
                    s_spin[sl][wid] += g_spu_spin_polls - _spin0;
                    if (k_part == 0 && t1.QuadPart - s_last_rep[sl] > 10 * s_freq2) {
                        double win = (double)(t1.QuadPart - s_last_rep[sl]) / (double)s_freq2;
                        s_last_rep[sl] = t1.QuadPart;
                        spurs_wkl_quiet_report(sl, ea, s_runs[sl], s_busy[sl], win);   /* v102 [WKL-QUIET] */
                        fprintf(stderr, "[pm-time] inst %d, last %.1fs (%u threads, %s):", sl, win, k_nparts, si->solo ? "solo+wid%n" : "wid%n");
                        for (int w2 = 0; w2 < 16; w2++) if (s_runs[sl][w2]) {
                            fprintf(stderr, " | w%d t%u %.0fms(%.0f%%) %ur/%ub max%.1fms spin%lluk", w2, spurs_wkl_part((u32)w2, k_nparts, si->solo),
                                    1000.0 * s_tsum[sl][w2] / s_freq2, 100.0 * s_tsum[sl][w2] / s_freq2 / win,
                                    s_runs[sl][w2], s_busy[sl][w2], 1000.0 * s_tmax[sl][w2] / s_freq2,
                                    s_spin[sl][w2] / 1000ull);
                            s_tsum[sl][w2] = 0; s_tmax[sl][w2] = 0; s_runs[sl][w2] = 0; s_busy[sl][w2] = 0; s_spin[sl][w2] = 0;
                        }
                        fprintf(stderr, "%c", 10);
                        if (sl == 0) spurs_lanes_report(win);
                    }
                }
            }
            /* "Found work" heuristic: a module that did something polls the
             * kernel for MORE work before exiting (selectWorkload calls >0);
             * an idle module exits immediately with polls==0. Grow the idle
             * streak on the latter, reset on the former. */
            if (g_spurs_pm_polls == 0) { if (s_idle[wid] < 8) s_idle[wid]++; }
            else { s_idle[wid] = 0; k_busy_pass = 1; }
            spurs_stall_post((int)(si - s_inst), wid, g_spurs_pm_polls != 0);   /* v97b */
            /* v31d SPURS_STALL_MS (default 500; 0 = off): stall breaker. Freezes since v31:
             * the main thread waits on a CTaskMgr list (w8, w10) that still holds 7
             * unclaimed jobs while EVERY lane of that workload reports no work, forever
             * (ROWDUMP: progress 0x39/0x3B vs target 0x40/0x42). The PM caches its view
             * of the queue line in LS; a missed line-changed notification would leave it
             * blind. A workload that has found nothing for SPURS_STALL_MS gets its lanes
             * restarted fresh on their next run (the same reset a list switch does), at
             * most once per SPURS_STALL_MS. Harmless for a truly idle workload. */
            {
                static int s_sms = -1; static LONGLONG s_sfreq = 0;
                if (s_sms < 0) { const char* e = getenv("SPURS_STALL_MS"); s_sms = e ? atoi(e) : 500;
                    LARGE_INTEGER f; QueryPerformanceFrequency(&f); s_sfreq = f.QuadPart;
                    fprintf(stderr, "[spurs-kern] stall breaker: %d ms (v31d)%c", s_sms, 10); }
                if (s_sms > 0) {
                    static LONGLONG s_last_busy[MAX_SPURS_INST][16], s_last_brk[MAX_SPURS_INST][16];
                    static long s_brk_n = 0, s_rev_n = 0;
                    static unsigned char s_armed[MAX_SPURS_INST][16];
                    static unsigned char s_noted[MAX_SPURS_INST][16];   /* v97b: skip already logged this idle stretch */
                    static long s_skip_n = 0;
                    int sl2 = (int)(si - s_inst);
                    LARGE_INTEGER tn; QueryPerformanceCounter(&tn);
                    LONGLONG lim = (LONGLONG)s_sms * s_sfreq / 1000;
                    if (wid < 16) {
                        if (g_spurs_pm_polls) {
                            if (s_armed[sl2][wid]) {   /* first busy run after a break */
                                s_armed[sl2][wid] = 0; s_rev_n++;
                                if (s_rev_n <= 200 || (s_rev_n % 100) == 0)
                                    fprintf(stderr, "[spurs-kern] stall breaker: inst %d wid %u busy again %.0f ms after a restart (#%ld; often just new work)%c",
                                            sl2, wid, 1000.0 * (double)(tn.QuadPart - s_last_brk[sl2][wid]) / (double)s_sfreq, s_rev_n, 10);
                            }
                            s_noted[sl2][wid] = 0;
                            s_last_busy[sl2][wid] = tn.QuadPart;
                        } else {
                            if (!s_last_busy[sl2][wid]) s_last_busy[sl2][wid] = tn.QuadPart;
                            if (tn.QuadPart - s_last_busy[sl2][wid] > lim && tn.QuadPart - s_last_brk[sl2][wid] > lim) {
                                /* v97b: restart only on evidence of a stall (see spurs_stall_pre). */
                                int mode = spurs_stall_mode();
                                long long now_us = tw_now_us(), tw0 = g_spurs_tw_open_us;
                                long long pend = s_stl_pend_us[sl2][wid];
                                int ext = pend != 0;
                                int twl = tw0 && now_us - tw0 > (long long)s_sms * 1000;
                                int twh = tw0 && now_us - tw0 > s_stall_hard_us;   /* v100: a real deadlock only */
                                int doit = mode == 0 || (mode == 1 && twl) || (mode == 3 && (ext || twl)) || (mode == 4 && twh);   /* v97b2, v100 */
                                double idle_ms = 1000.0 * (double)(tn.QuadPart - s_last_busy[sl2][wid]) / (double)s_sfreq;
                                s_last_brk[sl2][wid] = tn.QuadPart;
                                if (doit) {
                                    extern void spurs_pm_force_fresh(uint32_t);
                                    spurs_pm_force_fresh(wid);
                                    s_armed[sl2][wid] = 1; s_brk_n++;
                                    if (s_brk_n <= 200 || (s_brk_n % 100) == 0)
                                        fprintf(stderr, "[STALL] t=%.1fs inst %d wid %u idle %.0f ms, queue line changed %ld x (first %.0f ms ago), PPU wait open %.0f ms -> fresh lanes (#%ld, mode %d)\n",
                                                stl_t(), sl2, wid, idle_ms, s_stl_chg[sl2][wid], ext ? (now_us - pend) / 1000.0 : 0.0,
                                                tw0 ? (now_us - tw0) / 1000.0 : 0.0, s_brk_n, mode);
                                } else if (!s_noted[sl2][wid]) {
                                    s_noted[sl2][wid] = 1; s_skip_n++;
                                    if (s_skip_n <= 200 || (s_skip_n % 100) == 0)
                                        fprintf(stderr, "[STALL] t=%.1fs inst %d wid %u idle %.0f ms, queue line changed %ld x, PPU wait open %.0f ms -> %s (#%ld; old v31d would restart its lanes)\n",
                                                stl_t(), sl2, wid, idle_ms, s_stl_chg[sl2][wid], tw0 ? (now_us - tw0) / 1000.0 : 0.0,
                                                mode == 2 ? "log only (SPURS_STALL_MODE=2)" : (mode == 4 && twl) ? "no restart (v100: PPU wait < SPURS_STALL_HARD_MS)" : "no restart (no stuck PPU wait)", s_skip_n);
                                    if (!tw0) { extern void ps3_freeze_note_spurs_idle(int, unsigned); ps3_freeze_note_spurs_idle(sl2, wid); }   /* v98b FREEZE */
                                }
                            }
                        }
                    }
                }
            }
            }
        }
    }
}

s32 cellSpursReadyCountStore(CellSpurs* spurs, CellSpursWorkloadId wid,
                             u32 value)
{
    if (!spurs) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (wid >= CELL_SPURS_MAX_WORKLOAD) return CELL_SPURS_CORE_ERROR_INVAL;
    if (!s_workloads[wid].in_use) return CELL_SPURS_CORE_ERROR_SRCH;

    s_workloads[wid].readyCount = value;
    /* The real store: the instance byte the kernel (poll thread) watches. */
    *(vm_base + (u32)(uintptr_t)spurs + SPURS_WKL_READY1 + wid) = (u8)value;
    {   static int _n = 0;
        if (_n < 32)
            printf("[cellSpurs] ReadyCountStore(wid=%u, value=%u)\n", wid, value);
        else if (_n == 32)
            printf("[cellSpurs] ReadyCountStore further logs suppressed\n");
        _n++;
    }
    return CELL_OK;
}

s32 cellSpursReadyCountSwap(CellSpurs* spurs, CellSpursWorkloadId wid,
                            u32* old, u32 value)
{
    if (!spurs || !old) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (wid >= CELL_SPURS_MAX_WORKLOAD) return CELL_SPURS_CORE_ERROR_INVAL;
    if (!s_workloads[wid].in_use) return CELL_SPURS_CORE_ERROR_SRCH;

    vm_write32((u32)(uintptr_t)old, s_workloads[wid].readyCount);
    s_workloads[wid].readyCount = value;
    return CELL_OK;
}

s32 cellSpursReadyCountCompareAndSwap(CellSpurs* spurs,
                                       CellSpursWorkloadId wid,
                                       u32* old, u32 compare, u32 value)
{
    if (!spurs || !old) return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (wid >= CELL_SPURS_MAX_WORKLOAD) return CELL_SPURS_CORE_ERROR_INVAL;
    if (!s_workloads[wid].in_use) return CELL_SPURS_CORE_ERROR_SRCH;

    vm_write32((u32)(uintptr_t)old, s_workloads[wid].readyCount);
    if (s_workloads[wid].readyCount == compare)
        s_workloads[wid].readyCount = value;

    return CELL_OK;
}

s32 cellSpursWakeUp(CellSpurs* spurs)
{
    if (!spurs) return CELL_SPURS_CORE_ERROR_NULL_POINTER;

    /* Start the SPU program of every runnable workload on this instance.
     *
     * cellSpursAddWorkload only REGISTERS the policy module -- it publishes the
     * workload into the BE instance and returns. WakeUp is what the guest
     * expects to actually put it on an SPU, and this was a stub returning
     * CELL_OK, so a title whose work is a WORKLOAD (rather than a taskset task,
     * which cellSpursCreateTask does dispatch) never ran any SPU code at all.
     *
     * Guitar Hero III is one: its job library does AddWorkload(pm, 6720 bytes)
     * -> SetExceptionEventHandler -> ReadyCountStore(8) -> WakeUp, and the
     * decompression jobs it then queues are picked up by that program. With
     * WakeUp inert the jobs were submitted, counted, and never executed or
     * retired -- the PPU blocked forever on a counter nothing could decrement,
     * while the only SPU images running were the two taskset tasks.
     *
     * Dispatch is async for the same reason CreateTask uses it: a workload
     * policy module is a persistent worker, and running it inline would block
     * this PPU thread forever. Guarded by `started` so a second WakeUp (they
     * are common -- the guest calls it whenever it adds work) does not spawn
     * another copy. */
    const uint32_t spurs_ea = (uint32_t)(uintptr_t)spurs;
    for (u32 i = 0; i < CELL_SPURS_MAX_WORKLOAD; i++) {
        SpursWorkload* w = &s_workloads[i];
        if (!w->in_use || w->started || w->spurs_ea != spurs_ea) continue;
        if (!w->pm || !w->sizePm) continue;
        const uint8_t* host = GUEST_PTR(w->pm, const uint8_t*);
        if (!host) continue;
        w->started = 1;
        fprintf(stderr, "[cellSpurs] WakeUp: starting workload %u pm=0x%08X size=%u\n",
                i, (uint32_t)(uintptr_t)w->pm, w->sizePm);
        fflush(stderr);
        /* One instance. readyCount is how many SPUs the workload MAY run on,
         * and running 2 or 4 changes nothing for the title this was written
         * against -- measured, not assumed -- so the conservative reading
         * stands until a title needs otherwise. */
        spu_workload_dispatch_async(host, w->sizePm, (uint32_t)w->data);
    }
    return CELL_OK;
}

/* =========================================================================
 * Event flags
 * =====================================================================*/

s32 cellSpursEventFlagInitialize(CellSpursTaskset* taskset,
                                 CellSpursEventFlag* eventFlag,
                                 u32 clearMode, u32 direction)
{
    /* All pointers are raw guest EAs; the flag lives in guest memory in the
     * REAL BE layout (the SPU task library DMAs this exact struct). */
    uint32_t eventFlag_ea = (uint32_t)(uintptr_t)eventFlag;
    uint32_t taskset_ea   = (uint32_t)(uintptr_t)taskset;

    if (!eventFlag_ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    if (eventFlag_ea & 0x7F)
        return CELL_SPURS_TASK_ERROR_ALIGN;

    /* Re-initialization: recycle the sync slot. */
    EventFlagSync* old = ef_sync_find(eventFlag_ea);
    if (old) ef_sync_free(old);

    for (uint32_t o = 0; o < EF_GUEST_SIZE; o += 8)
        vm_write64(eventFlag_ea + o, 0);
    vm_write8(eventFlag_ea + EF_DIRECTION,  (uint8_t)direction);
    vm_write8(eventFlag_ea + EF_CLEAR_MODE, (uint8_t)clearMode);
    /* addr = the owning taskset (isIwl=0); SPU-side Set uses it to find whom
     * to signal, and our Set hook passes it to spu_taskset_signal_task. */
    vm_write8(eventFlag_ea + EF_IS_IWL, 0);
    vm_write64(eventFlag_ea + EF_ADDR, (uint64_t)taskset_ea);

    EventFlagSync* sync = ef_sync_alloc(eventFlag_ea);
    if (!sync) {
        printf("[cellSpurs] EventFlagInitialize: no free sync slots!\n");
        return CELL_SPURS_TASK_ERROR_NOMEM;
    }

    printf("[cellSpurs] EventFlagInitialize(clearMode=%u, direction=%u) flagEA=0x%08X taskset=0x%08X\n",
           clearMode, direction, eventFlag_ea, taskset_ea);
    return CELL_OK;
}

s32 cellSpursEventFlagAttachLv2EventQueue(CellSpursEventFlag* eventFlag)
{
    eventFlag = GUEST_PTR(eventFlag, CellSpursEventFlag*);
    if (!eventFlag) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    printf("[cellSpurs] EventFlagAttachLv2EventQueue()\n");
    return CELL_OK;
}

s32 cellSpursEventFlagDetachLv2EventQueue(CellSpursEventFlag* eventFlag)
{
    eventFlag = GUEST_PTR(eventFlag, CellSpursEventFlag*);
    if (!eventFlag) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    printf("[cellSpurs] EventFlagDetachLv2EventQueue()\n");
    return CELL_OK;
}

s32 cellSpursEventFlagSet(CellSpursEventFlag* eventFlag, u16 bits)
{
    uint32_t ea = (uint32_t)(uintptr_t)eventFlag;
    if (!ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    EventFlagSync* sync = ef_sync_get(ea);
    if (!sync)
        return CELL_SPURS_TASK_ERROR_STAT;

    { static int _n=0; if (_n++ < 40 || (_n%1000)==0)
        fprintf(stderr, "[cellSpurs] EventFlagSet#%d flagEA=0x%08X bits=0x%04X "
                "events=0x%04X used=0x%04X pend=0x%04X mode=0x%04X\n",
                _n, ea, (unsigned)bits,
                vm_read16(ea + EF_EVENTS), vm_read16(ea + EF_SPU_USED_SLOTS),
                vm_read16(ea + EF_SPU_PENDING_RECV), vm_read16(ea + EF_SPU_WAIT_MODE)); }

    ef_lock(sync);
    spurs_ef_set_locked(ea, bits);
    ef_broadcast(sync);
    ef_unlock(sync);

    return CELL_OK;
}

/* SPURS_EF_STATS=1: per flag, how many PPU waits completed in the last 5 s
 * and their mean / max duration -- one SPU task round trip each. */
static void ef_wait_stats(uint32_t ea, long long t0)
{
    static int on = -1; if (on < 0) on = getenv("SPURS_EF_STATS") ? 1 : 0;
    if (!on) return;
    static uint32_t fea[8]; static unsigned n[8]; static double sum[8], mx[8];
    static long long last; static SRWLOCK lk = SRWLOCK_INIT;
    LARGE_INTEGER now, f; QueryPerformanceCounter(&now); QueryPerformanceFrequency(&f);
    double us = (now.QuadPart - t0) * 1e6 / (double)f.QuadPart;
    AcquireSRWLockExclusive(&lk);
    for (int i = 0; i < 8; i++)
        if (fea[i] == ea || !fea[i]) { fea[i] = ea; n[i]++; sum[i] += us; if (us > mx[i]) mx[i] = us; break; }
    if (!last) last = now.QuadPart;
    if ((now.QuadPart - last) > f.QuadPart) {   /* v106: 1 s windows (was 5 s), [FREEZE] time base */
        extern double ps3_fz_t(void);
        for (int i = 0; i < 8 && fea[i]; i++) {
            if (!n[i]) continue;
            fprintf(stderr, "[ef-stats] t=%.1fs flag=0x%08X waits=%u (%.1f/s) mean=%.0fus max=%.0fus total=%.0fms%c",
                    ps3_fz_t(), fea[i], n[i], n[i] * (double)f.QuadPart / (now.QuadPart - last),
                    n[i] ? sum[i] / n[i] : 0.0, mx[i], sum[i] / 1000.0, 10);
            n[i] = 0; sum[i] = mx[i] = 0;
        }
        last = now.QuadPart;
    }
    ReleaseSRWLockExclusive(&lk);
}

s32 cellSpursEventFlagWait(CellSpursEventFlag* eventFlag, u16* bits,
                           u32 mode)
{
    uint32_t ea      = (uint32_t)(uintptr_t)eventFlag;
    uint32_t bits_ea = (uint32_t)(uintptr_t)bits;
    if (!ea || !bits_ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    EventFlagSync* sync = ef_sync_get(ea);
    if (!sync)
        return CELL_SPURS_TASK_ERROR_STAT;

    u16 pattern = vm_read16(bits_ea);
    if (ftl_line() && (ea & ~127u) == g_ftl_line) ps3_ftl(8, ea, pattern);   /* v115 */

    /* v111 SPURS_EF_SPIN_US (default 500; 0 = old): re-check the guest flag word at full speed for a short
     * while BEFORE taking the lock and falling into the 2 ms CV poll. The FIOS streaming path waits here once
     * per request, ~650-800 times/s while loading; play110: mean wait 1.2-1.7 ms while the FIOS SPU task sat
     * idle 68-95% of the time -> the request finished quickly but the PPU noticed it on its next 2 ms tick
     * (an SPU set that does not reach ef_line_committed, e.g. a plain DMA PUT, wakes nobody). Spinning
     * outside the lock keeps the SPU-side set paths (spurs_ef_set_from_spu takes ef_lock) unblocked. */
    {   static int spin_us = -1;
        if (spin_us < 0) { const char* e = getenv("SPURS_EF_SPIN_US"); spin_us = e ? atoi(e) : 500; if (spin_us < 0) spin_us = 0;
                           fprintf(stderr, "[cellSpurs] v111 EventFlagWait: spin up to %d us on the flag word before the CV wait (SPURS_EF_SPIN_US)\n", spin_us); }
        if (spin_us > 0) {
            LARGE_INTEGER sf, st0, st; QueryPerformanceFrequency(&sf); QueryPerformanceCounter(&st0);
            const long long lim = (long long)sf.QuadPart * spin_us / 1000000;
            for (unsigned it = 0;; it++) {
                u16 cur = vm_read16(ea + EF_EVENTS);
                if (mode == CELL_SPURS_EVENT_FLAG_AND ? ((cur & pattern) == pattern) : ((cur & pattern) != 0)) break;
                YieldProcessor();
                if ((it & 63) == 63) { QueryPerformanceCounter(&st); if (st.QuadPart - st0.QuadPart > lim) break; }
            }
        }
    }

    ef_lock(sync);

    /* Block until the requested bit pattern is satisfied in the GUEST struct
     * (BE events word at +0x00) — set by an SPU task (lifted code / taskset
     * syscall) or another PPU thread. Poll-based: ef_wait_timed ticks re-read
     * guest memory, so task-side PUTLLC stores are observed without needing
     * the real lv2-event-queue notification path.
     *
     * A wait the SPU never satisfies is a REAL deadlock and must present as
     * one — that is the signal naming the workload that is not executing.
     * SPURS_EF_FORCE=1 restores the old fake for A/B comparison only. */
    static int s_force = -1;
    if (s_force < 0) s_force = getenv("SPURS_EF_FORCE") ? 1 : 0;
    unsigned waits = 0;
    u16 current;
    __atomic_add_fetch(&s_ef_ppu_waiters, 1, __ATOMIC_SEQ_CST);
    LARGE_INTEGER ef_t0; QueryPerformanceCounter(&ef_t0);
    for (;;) {
        current = vm_read16(ea + EF_EVENTS);

        if (mode == CELL_SPURS_EVENT_FLAG_AND) {
            if ((current & pattern) == pattern)
                break;
        } else {
            /* OR mode: any requested bit set */
            if ((current & pattern) != 0)
                break;
        }

        if (!ef_wait_timed(sync, 2)) {
            /* Report the stall once per second, naming what would have to run. */
            if (++waits % 500 == 0) {
                static int _n = 0;
                if (_n < 24) { _n++;
                    fprintf(stderr, "[cellSpurs] EventFlagWait BLOCKED tid=%lu %us on pattern 0x%04X "
                                    "(mode=%s, bits=0x%04X) flagEA=0x%08X -- waiting for an SPU "
                                    "workload to cellSpursEventFlagSet it\n",
                            (unsigned long)GetCurrentThreadId(), waits / 500, pattern,
                            mode == CELL_SPURS_EVENT_FLAG_AND ? "AND" : "OR",
                            current, ea);
                    fflush(stderr);
                }
            }
            if (s_force && waits >= 1000) {
                fprintf(stderr, "[cellSpurs] EventFlagWait: SPURS_EF_FORCE -- faking pattern "
                                "0x%04X (NOT real; A/B only)\n", pattern);
                vm_write16(ea + EF_EVENTS, (u16)(current | pattern));
            }
        }
    }

    __atomic_sub_fetch(&s_ef_ppu_waiters, 1, __ATOMIC_SEQ_CST);
    if (g_ftl_line && g_ftl_line != 0xFFFFFFFFu && (ea & ~127u) == g_ftl_line) { ps3_ftl(9, ea, current); ps3_fiosreq_woke(ea, current); }   /* v115/v118 */
    ef_wait_stats(ea, ef_t0.QuadPart);
    { LARGE_INTEGER fn, ff; QueryPerformanceCounter(&fn); QueryPerformanceFrequency(&ff);   /* v113 FRAMEREC */
      g_fr_ef_us += (fn.QuadPart - ef_t0.QuadPart) * 1000000 / ff.QuadPart; g_fr_ef_n++; }
    /* Hand back the observed bits; consume the received ones on AUTO clear. */
    vm_write16(bits_ea, current);
    u16 received = (mode == CELL_SPURS_EVENT_FLAG_AND) ? pattern
                                                       : (u16)(current & pattern);
    { static int _n = 0; if (_n++ < 40)
        fprintf(stderr, "[cellSpurs] EventFlagWait WAKE tid=%lu flagEA=0x%08X "
                "pattern=0x%04X got=0x%04X (waits=%u)\n",
                (unsigned long)GetCurrentThreadId(), ea, pattern, current, waits); }
    if (vm_read8(ea + EF_CLEAR_MODE) == CELL_SPURS_EVENT_FLAG_CLEAR_AUTO) {
        /* Clear from the CURRENT word: bits an SPU set since `current` was
         * read must survive. */
        spu_lockline_lock();
        ef_w16(ea + EF_EVENTS, (u16)(ef_r16(ea + EF_EVENTS) & ~received));
        ef_line_end(ea);
    }

    ef_unlock(sync);

    return CELL_OK;
}

s32 cellSpursEventFlagTryWait(CellSpursEventFlag* eventFlag, u16* bits,
                              u32 mode)
{
    uint32_t ea      = (uint32_t)(uintptr_t)eventFlag;
    uint32_t bits_ea = (uint32_t)(uintptr_t)bits;
    if (!ea || !bits_ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    EventFlagSync* sync = ef_sync_get(ea);
    if (!sync)
        return CELL_SPURS_TASK_ERROR_STAT;

    u16 pattern = vm_read16(bits_ea);

    ef_lock(sync);

    u16 current = vm_read16(ea + EF_EVENTS);

    if (mode == CELL_SPURS_EVENT_FLAG_AND) {
        if ((current & pattern) != pattern) {
            ef_unlock(sync);
            return CELL_SPURS_TASK_ERROR_BUSY;
        }
    } else {
        if ((current & pattern) == 0) {
            ef_unlock(sync);
            return CELL_SPURS_TASK_ERROR_BUSY;
        }
    }

    vm_write16(bits_ea, current);
    u16 received = (mode == CELL_SPURS_EVENT_FLAG_AND) ? pattern
                                                       : (u16)(current & pattern);
    if (vm_read8(ea + EF_CLEAR_MODE) == CELL_SPURS_EVENT_FLAG_CLEAR_AUTO) {
        spu_lockline_lock();
        ef_w16(ea + EF_EVENTS, (u16)(ef_r16(ea + EF_EVENTS) & ~received));
        ef_line_end(ea);
    }

    ef_unlock(sync);
    return CELL_OK;
}

s32 cellSpursEventFlagClear(CellSpursEventFlag* eventFlag, u16 bits)
{
    uint32_t ea = (uint32_t)(uintptr_t)eventFlag;
    if (!ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    EventFlagSync* sync = ef_sync_get(ea);
    if (!sync)
        return CELL_SPURS_TASK_ERROR_STAT;

    ef_lock(sync);
    spu_lockline_lock();
    ef_w16(ea + EF_EVENTS, (u16)(ef_r16(ea + EF_EVENTS) & ~bits));
    ef_line_end(ea);
    ef_unlock(sync);

    return CELL_OK;
}

s32 cellSpursEventFlagGetDirection(CellSpursEventFlag* eventFlag,
                                   u32* direction)
{
    uint32_t ea     = (uint32_t)(uintptr_t)eventFlag;
    uint32_t dir_ea = (uint32_t)(uintptr_t)direction;
    if (!ea || !dir_ea)
        return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    vm_write32(dir_ea, vm_read8(ea + EF_DIRECTION));
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Additional functions needed by Tokyo Jungle (from RPCS3 audit)
 * -----------------------------------------------------------------------*/

/* _cellSpursEventFlagInitialize — internal init with more parameters */
s32 _cellSpursEventFlagInitialize(void* spurs, void* taskset,
                                    CellSpursEventFlag* eventFlag,
                                    u32 clearMode, u32 direction)
{
    (void)spurs; (void)taskset;
    printf("[cellSpurs] _EventFlagInitialize(clearMode=%u, dir=%u)\n",
           clearMode, direction);
    if (!eventFlag) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    /* Forward raw guest pointers; cellSpursEventFlagInitialize translates them
     * (translating here too would double-translate -> out-of-bounds). */
    return cellSpursEventFlagInitialize((CellSpursTaskset*)taskset, eventFlag, clearMode, direction);
}

/* _cellSpursSendSignal — internal signal delivery */
s32 _cellSpursSendSignal(void* taskset, u32 taskId)
{
    uint32_t taskset_ea = (uint32_t)(uintptr_t)taskset;
    printf("[cellSpurs] _SendSignal(taskset=0x%08X id=%u)\n", taskset_ea, taskId);
    /* Real delivery now that SPU tasks execute (comment was stale). */
    if (taskset_ea) spu_taskset_signal_task(taskset_ea, taskId);
    return CELL_OK;
}

/* =========================================================================
 * Job chains (LBP's render path: the game emits SPURS job descriptors and
 * chains them via u64 command words; the jobchain policy module walks the
 * chain on SPU). Facts-first bring-up: decode + log the REAL BE guest
 * structures at the SDK offsets; execution wiring lands once the logged
 * shapes confirm the descriptor formats.
 * =====================================================================*/

/* Real (BE) CellSpursJobChainAttribute offsets. */
enum {
    JCATTR_REVISION   = 0x00,   /* be u32 */
    JCATTR_SDKVERSION = 0x04,   /* be u32 */
    JCATTR_ENTRY      = 0x08,   /* be u32: EA of the first jobchain command word */
    JCATTR_SIZE_DESC  = 0x0C,   /* be u16: sizeJobDescriptor */
    JCATTR_MAX_GRAB   = 0x0E,   /* be u16: maxGrabbedJob */
    JCATTR_PRIORITY   = 0x10,   /* u8[8] */
    JCATTR_MAX_CONT   = 0x18,   /* be u32 */
    JCATTR_AUTO_RDY   = 0x1C,   /* u8 bool: autoReadyCount */
    JCATTR_TAG1       = 0x20,   /* be u32 */
    JCATTR_TAG2       = 0x24,   /* be u32 */
    JCATTR_FIXED_MEM  = 0x28,   /* u8 bool */
    JCATTR_MAX_SIZE_D = 0x2C,   /* be u32 */
    JCATTR_INIT_SPU   = 0x30,   /* be u32 */
    JCATTR_NAME       = 0x34,   /* be u32: char* EA (SetName) */
};

/* Host-side jobchain registry (the CellSpursJobChain guest struct is opaque
 * to the game; we track what we need beside it). */
#define MAX_JOBCHAINS 32
static struct {
    u32 jc_ea;        /* CellSpursJobChain EA (0 = free) */
    u32 entry_ea;     /* first command word EA */
    u16 size_desc;
    u16 max_grab;
    int run_count;
    volatile long running;   /* 1 while a host thread walks this chain */
} s_jobchains[MAX_JOBCHAINS];

static void jc_dump_commands(const char* tag, u32 ea, int max_words)
{
    printf("[cellSpurs] %s chain@0x%08X commands:", tag, ea);
    for (int i = 0; i < max_words; i++) {
        u64 cmd = vm_read64(ea + (u32)i * 8);
        printf("\n    [%2d] 0x%016llX", i, (unsigned long long)cmd);
        if (cmd == 0) { printf(" (halt/empty)"); break; }
    }
    printf("\n");
}

/* SDK ABI (cell/spurs/job_chain.h): the REVISIONS come first -- attr is r5.
 *   _cellSpursJobChainAttributeInitialize(jmRevision, sdkRevision, attr,
 *       jobChainEntry, sizeJobDescriptor, maxGrabbedJob, priorityTable,
 *       maxContention, [stack: autoRequestSpuCount, tag1, tag2,
 *       isFixedMemAlloc, maxSizeJobDescriptor, initialRequestSpuCount]) */
s32 _cellSpursJobChainAttributeInitialize(u32 jmRevision, u32 sdkRevision, u64 attr_ea,
                                          u64 entry_ea, u32 sizeJobDescriptor,
                                          u32 maxGrabbedJob, u64 prio_ea, u32 maxContention)
{
    if (!attr_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    vm_write32((u32)attr_ea + JCATTR_REVISION,   jmRevision);
    vm_write32((u32)attr_ea + JCATTR_SDKVERSION, sdkRevision);
    vm_write32((u32)attr_ea + JCATTR_ENTRY,      (u32)entry_ea);
    vm_write32((u32)attr_ea + JCATTR_SIZE_DESC,
               ((sizeJobDescriptor & 0xFFFFu) << 16) | (maxGrabbedJob & 0xFFFFu));
    for (int i = 0; i < 8; i++)
        *(vm_base + (u32)attr_ea + JCATTR_PRIORITY + i) =
            prio_ea ? *(vm_base + (u32)prio_ea + i) : 0;
    vm_write32((u32)attr_ea + JCATTR_MAX_CONT, maxContention);
    /* args 9+ (autoReadyCount, tag1, tag2, isFixedMemAlloc, maxSizeJobDescriptor,
     * initSpuCount) are on the guest stack, beyond the 8-GPR HLE adapter -- defaults. */
    vm_write32((u32)attr_ea + JCATTR_AUTO_RDY,   0);
    vm_write32((u32)attr_ea + JCATTR_TAG1,       0);
    vm_write32((u32)attr_ea + JCATTR_TAG2,       0);
    vm_write32((u32)attr_ea + JCATTR_FIXED_MEM,  0);
    vm_write32((u32)attr_ea + JCATTR_MAX_SIZE_D, 0);
    vm_write32((u32)attr_ea + JCATTR_INIT_SPU,   0);
    vm_write32((u32)attr_ea + JCATTR_NAME,       0);
    printf("[cellSpurs] _JobChainAttributeInitialize(attr=0x%08X entry=0x%08X sizeDesc=%u maxGrab=%u maxCont=%u)\n",
           (u32)attr_ea, (u32)entry_ea, sizeJobDescriptor, maxGrabbedJob, maxContention);
    return CELL_OK;
}

s32 cellSpursJobChainAttributeSetName(u64 attr_ea, u64 name_ea)
{
    if (!attr_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    vm_write32((u32)attr_ea + JCATTR_NAME, (u32)name_ea);
    printf("[cellSpurs] JobChainAttributeSetName(attr=0x%08X, \"%s\")\n",
           (u32)attr_ea, name_ea ? (const char*)(vm_base + (u32)name_ea) : "-");
    return CELL_OK;
}

/* Register (or re-register) a job chain handle. Shared by both Create forms. */
static s32 jc_register(u32 jc_ea, u32 entry_ea, u16 size_desc, u16 max_grab,
                       const char* who)
{
    for (int i = 0; i < MAX_JOBCHAINS; i++) {
        if (!s_jobchains[i].jc_ea || s_jobchains[i].jc_ea == jc_ea) {
            s_jobchains[i].jc_ea     = jc_ea;
            s_jobchains[i].entry_ea  = entry_ea;
            s_jobchains[i].size_desc = size_desc;
            s_jobchains[i].max_grab  = max_grab;
            s_jobchains[i].run_count = 0;
            return CELL_OK;
        }
    }
    printf("[cellSpurs] %s: registry full\n", who);
    return CELL_OK;
}

s32 cellSpursCreateJobChainWithAttribute(u64 spurs_ea, u64 jc_ea, u64 attr_ea)
{
    if (!spurs_ea || !jc_ea || !attr_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    u32 entry   = vm_read32((u32)attr_ea + JCATTR_ENTRY);
    u32 sd_mg   = vm_read32((u32)attr_ea + JCATTR_SIZE_DESC);
    u32 name_ea = vm_read32((u32)attr_ea + JCATTR_NAME);

    {   /* Layout ground truth (same rationale as AddWorkloadWA). */
        static int _n = 0;
        if (_n < 3) {
            printf("[cellSpurs] CreateJobChainWA jc=0x%08X attr=0x%08X raw:", (u32)jc_ea, (u32)attr_ea);
            for (int o = 0; o < 0x40; o += 4) {
                if ((o & 15) == 0) printf("\n    +%02X:", o);
                printf(" %08X", vm_read32((u32)attr_ea + o));
            }
            printf("\n");
            if (entry && entry < 0x10000000u) jc_dump_commands("CreateJobChainWA", entry, 16);
        } else if (_n == 3) {
            printf("[cellSpurs] CreateJobChainWA raw dumps suppressed from here\n");
        }
        _n++;
    }
    printf("[cellSpurs] CreateJobChainWithAttribute(jc=0x%08X entry=0x%08X sizeDesc=%u maxGrab=%u name=\"%s\")\n",
           (u32)jc_ea, entry, sd_mg >> 16, sd_mg & 0xFFFFu,
           name_ea ? (const char*)(vm_base + name_ea) : "-");

    return jc_register((u32)jc_ea, entry, (u16)(sd_mg >> 16), (u16)(sd_mg & 0xFFFFu),
                       "CreateJobChainWithAttribute");
}

/* cellSpursCreateJobChain -- the no-attribute form, and the one the WWS job
 * manager's own wrapper calls. Same registration, with the fields passed
 * directly instead of read back out of a CellSpursJobChainAttribute:
 *
 *   cellSpursCreateJobChain(CellSpurs*, CellSpursJobChain*, const u64* entry,
 *                           u16 sizeJobDescriptor, u16 maxGrabbedJob,
 *                           const u8 priorityTable[8], u32 maxContention,
 *                           bool autoRequestSpuCount, u32 tag1, u32 tag2)
 *
 * tag1/tag2 are guest-stack args, past the 8-GPR HLE adapter; nothing here
 * needs them. Without this entry point the chain is never registered at all,
 * so the later kick finds "UNKNOWN chain (no Create seen)" and returns
 * CELL_OK having walked nothing -- and the title waits forever on SPU work
 * that was never started. Gran Turismo 5 Prologue parks its whole boot there. */
s32 cellSpursCreateJobChain(u64 spurs_ea, u64 jc_ea, u64 entry_ea,
                            u32 sizeJobDescriptor, u32 maxGrabbedJob,
                            u64 prio_ea, u32 maxContention,
                            u32 autoRequestSpuCount)
{
    (void)spurs_ea; (void)prio_ea;
    if (!jc_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    static int _n = 0;
    if (_n++ < 3) {
        printf("[cellSpurs] CreateJobChain(jc=0x%08X entry=0x%08X sizeDesc=%u maxGrab=%u "
               "maxCont=%u autoReq=%u)\n",
               (u32)jc_ea, (u32)entry_ea, sizeJobDescriptor, maxGrabbedJob,
               maxContention, autoRequestSpuCount);
        if (entry_ea && (u32)entry_ea < 0x10000000u)
            jc_dump_commands("CreateJobChain", (u32)entry_ea, 16);
    }
    return jc_register((u32)jc_ea, (u32)entry_ea, (u16)sizeJobDescriptor,
                       (u16)maxGrabbedJob, "CreateJobChain");
}

/* ---------------------------------------------------------------------------
 * Job-chain execution
 *
 * A job chain is a u64 command stream (SDK cell/spurs/job_commands.h): the low
 * 3 bits select the opcode, and a nonzero word whose low 3 bits are 0 IS a job
 * pointer. The real jobchain policy module walks this on an SPU, fetching each
 * CellSpursJobHeader and running its binary.
 *
 * LBP's draw pipeline is built on this -- the jobs emit the GCM commands. With
 * the chain unimplemented the FIFO starves, the RSX `ref` fence stops advancing
 * and the game spins on it forever (the boot hang: ref frozen at 0x2B3 while
 * the chain sat un-run).
 *
 * We walk the stream on a host thread (one per chain; the real thing is async
 * on SPUs, so RunJobChain must not block the PPU) and push each job binary
 * through the same fingerprint -> lifted-SPU dispatch that already runs this
 * title's workload images.
 * -----------------------------------------------------------------------*/
enum {                              /* CellSpursJobHeader (48 B) */
    JH_EA_BINARY     = 0x00,        /* be u64: job binary EA (low 3 bits = flags) */
    JH_SIZE_BINARY   = 0x08,        /* be u16: binary size >> 4                   */
    JH_JOB_TYPE      = 0x2C,        /* u8                                         */
    JH_SIZE          = 0x30,
};

static void jc_run_one_job(u32 job_ea, int idx, u32 size_desc)
{
    /* Job descriptors are commonly recycled in place. Snapshot the descriptor
     * and its binary before fingerprinting: another chain/PPU producer may
     * rewrite job_ea while this synchronous host walker is staging the job. */
    u32 desc_bytes = size_desc ? size_desc : 256u;
    if (desc_bytes > 0x400u) desc_bytes = 0x400u;
    u8 desc_snapshot[0x400];
    memcpy(desc_snapshot, vm_base + job_ea, desc_bytes);
#define JC_D32(o) (((u32)desc_snapshot[(o)] << 24) | ((u32)desc_snapshot[(o)+1] << 16) | \
                   ((u32)desc_snapshot[(o)+2] << 8) | (u32)desc_snapshot[(o)+3])
    u64 ea_bin_raw = ((u64)JC_D32(JH_EA_BINARY) << 32) | JC_D32(JH_EA_BINARY + 4);
    u32 ea_bin     = (u32)(ea_bin_raw & ~7ull);          /* low 3 bits = flags */
    u32 size_bin   = ((JC_D32(JH_SIZE_BINARY) >> 16) & 0xFFFFu) << 4;
    u8  job_type   = desc_snapshot[JH_JOB_TYPE];

    { static int _n = 0;
      if (_n < 8) {
          printf("[cellSpurs]   job[%d] @0x%08X eaBinary=0x%08X size=%u type=0x%02X hdr:",
                 idx, job_ea, ea_bin, size_bin, job_type);
          for (int o = 0; o < JH_SIZE; o += 4) {
              if ((o & 15) == 0) printf("\n      +%02X:", o);
              printf(" %08X", JC_D32(o));
          }
          printf("\n");
      }
      _n++; }

    if (!ea_bin || !size_bin || ea_bin >= 0x10000000u) {
        static int _b = 0;
        if (_b++ < 4)
            printf("[cellSpurs]   job[%d]: implausible binary (ea=0x%08X size=%u) -- skipped\n",
                   idx, ea_bin, size_bin);
        return;
    }
    /* A jobchain job is NOT a SPURS task: Sony's jm2 stages the whole working
     * set in local store and enters the job's CRT with r3 = CellSpursJobContext2*
     * and r4 = the job descriptor, both LS pointers. Dispatching it on the task
     * ABI (arg EA in r3) left it reading a zeroed context and parking with no
     * DMA traffic at all. spu_workload_dispatch_job reproduces jm2's staging.
     * Synchronous by design: a job runs to completion, and the chain's own
     * NEXT/CALL/RET commands are what order them. */
    u8* image_snapshot = (u8*)malloc(size_bin);
    if (!image_snapshot) return;
    memcpy(image_snapshot, vm_base + ea_bin, size_bin);
    spu_workload_dispatch_job(image_snapshot, size_bin, job_ea,
                              desc_snapshot, desc_bytes);
    free(image_snapshot);
#undef JC_D32
}

/* Walk one chain's command stream. Bounded: a malformed or self-looping stream
 * must not spin a host thread forever. */
/* ---------------------------------------------------------------------------
 * Job guards
 *
 * A job chain can begin with a GUARD command (op 7, ext 7|(1<<3)), whose EA is
 * the CellSpursJobGuard. The chain BLOCKS there until the guard's notify count
 * reaches zero; the PPU releases it with cellSpursJobGuardNotify once the job's
 * parameters are ready. With autoReset the count is restored afterwards, so a
 * looping chain waits again on the next pass.
 *
 * These were unimplemented, and the walker fell through the GUARD command into
 * the default (ignore) case. Tokyo Jungle's "soc-job" chain is
 *
 *     [0] GUARD 0x02932C00      [1] JOB 0x02932D00
 *     [2] SYNC                  [3] NEXT -> [0]        (a loop)
 *
 * so the job ran immediately, on the first pass, against a parameter block the
 * game had not filled in yet -- 0x02937580 read back as all zeroes -- and the
 * SPU went on to compute a wild DMA address from it. Honouring the guard is
 * what makes "run this job when I say so" mean anything.
 * -----------------------------------------------------------------------*/

#define MAX_JOBGUARDS 32
static struct {
    u32 ea;          /* guest CellSpursJobGuard address (128-byte aligned) */
    u32 count;       /* notifications still outstanding                    */
    u32 reset;       /* value to restore when autoReset is set             */
    u32 pending;     /* notifications received, not yet consumed        */
    u32 auto_reset;
    int active;
} s_jobguards[MAX_JOBGUARDS];

static int jg_find(u32 ea)
{
    for (int i = 0; i < MAX_JOBGUARDS; i++)
        if (s_jobguards[i].active && s_jobguards[i].ea == ea) return i;
    return -1;
}

/* cellSpursJobGuardInitialize(CellSpursJobChain* jobChain,
 *                             CellSpursJobGuard* jobGuard,
 *                             u32 notifyCount, u8 requestSpuCount, u8 autoReset)
 * Argument order confirmed against the guest: r3 is the chain (0x02932A80,
 * the handle CreateJobChainWithAttribute registered) and r4 is the guard
 * (0x02932C00, the EA the chain's GUARD command carries). */
s32 cellSpursJobGuardInitialize(u64 jc_ea, u64 guard_ea, u32 notify_count,
                                u32 request_spu_count, u32 auto_reset)
{
    (void)jc_ea; (void)request_spu_count;
    u32 ea = (u32)guard_ea;
    if (!ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    int i = jg_find(ea);
    if (i < 0)
        for (i = 0; i < MAX_JOBGUARDS; i++) if (!s_jobguards[i].active) break;
    if (i >= MAX_JOBGUARDS) return CELL_SPURS_TASK_ERROR_AGAIN;

    s_jobguards[i].ea         = ea;
    s_jobguards[i].count      = notify_count;
    s_jobguards[i].reset      = notify_count;
    s_jobguards[i].pending    = 0;
    s_jobguards[i].auto_reset = auto_reset;
    s_jobguards[i].active     = 1;

    /* Mirror the count into guest memory. The SPU side reads the guard on
     * hardware; keeping the two consistent costs nothing and makes a guest-side
     * peek meaningful. */
    vm_write32(ea, notify_count);

    printf("[cellSpurs] JobGuardInitialize(guard=0x%08X chain=0x%08X notify=%u "
           "autoReset=%u)\n", ea, (u32)jc_ea, notify_count, auto_reset);
    return CELL_OK;
}

/* cellSpursJobGuardNotify(CellSpursJobGuard* jobGuard) -- one argument. */
s32 cellSpursJobGuardNotify(u64 guard_ea)
{
    u32 ea = (u32)guard_ea;
    int i = jg_find(ea);
    if (i < 0) {
        printf("[cellSpurs] JobGuardNotify(0x%08X) -- unknown guard\n", ea);
        return CELL_SPURS_TASK_ERROR_INVAL;
    }
    /* Count notifications as CREDITS rather than decrementing to a floor of
     * zero. The guard used to reload its count at the moment the walker
     * OBSERVED zero, which threw away any notify that arrived while the job
     * was still running -- the title notified 6 times and the chain completed
     * 2 laps, so the thread waiting on the other 4 completions never woke. */
    s_jobguards[i].pending++;
    { u32 remaining = (s_jobguards[i].pending >= s_jobguards[i].reset)
                    ? 0u : s_jobguards[i].reset - s_jobguards[i].pending;
      s_jobguards[i].count = remaining;
      vm_write32(ea, remaining); }
    { static int _n = 0;
      if (_n++ < 8)
          printf("[cellSpurs] JobGuardNotify(0x%08X) -> %u remaining\n",
                 ea, s_jobguards[i].count); }
    return CELL_OK;
}

/* cellSpursJobGuardReset(CellSpursJobGuard*) -- restore the initial count. */
s32 cellSpursJobGuardReset(u64 guard_ea)
{
    u32 ea = (u32)guard_ea;
    int i = jg_find(ea);
    if (i < 0) return CELL_SPURS_TASK_ERROR_INVAL;
    s_jobguards[i].count = s_jobguards[i].reset;
    vm_write32(ea, s_jobguards[i].count);
    return CELL_OK;
}

/* Wait for a guard to be released. Returns 1 if it opened, 0 on timeout (the
 * chain then gives up this pass rather than spinning the walker thread
 * forever). An unknown guard opens immediately -- a chain guarded by something
 * we never saw initialised must not deadlock the walker. */
static int jg_wait(u32 ea)
{
    int i = jg_find(ea);
    if (i < 0) return 1;
    for (int spin = 0; spin < 20000; spin++) {     /* ~20 s at 1 ms */
        if (s_jobguards[i].pending >= s_jobguards[i].reset) {
            /* Consume exactly one release worth of credits, so notifies that
             * arrived during the previous lap still count. */
            s_jobguards[i].pending -= s_jobguards[i].reset;
            if (!s_jobguards[i].auto_reset) s_jobguards[i].reset = 0;
            s_jobguards[i].count = s_jobguards[i].reset;
            vm_write32(ea, s_jobguards[i].count);
            return 1;
        }
#ifdef _WIN32
        Sleep(1);
#else
        usleep(1000);
#endif
    }
    printf("[cellSpurs] guard 0x%08X still closed after 20 s -- chain gives up\n", ea);
    return 0;
}

/* Tell the application a lap of chain work is done.
 *
 * Timing is the whole point here. The obvious place is after the walk
 * FINISHES, but a guarded chain never finishes: it loops back to its GUARD and
 * blocks for the next notify. Tokyo Jungle notifies the guard, then waits on
 * this queue, and would only notify again after the event arrives -- so
 * signalling at walker exit deadlocks both sides. Signal when the WORK
 * completes: at END, and on reaching a guard with jobs run this lap. */
#ifdef _WIN32
#  define SPURS_JOB_TLS __declspec(thread)
#else
#  define SPURS_JOB_TLS __thread
#endif
extern SPURS_JOB_TLS uint32_t g_spurs_job_mbox, g_spurs_job_mbox_intr, g_spurs_job_cmd;
extern SPURS_JOB_TLS int g_spurs_job_mbox_valid;

/* SPURS_EVENT_D3=<n>: probe. The value a job query returns to the PPU is read
 * from the completion event's data3 field (the guest stores r7 at sp+0xB8 and
 * reads the count back from sp+0xBC). This lets that be confirmed by observing
 * the returned count change, without having to guess the real value first. */
static int d3_probe_valid = 0;
static u64 spurs_event_data3_probe(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("SPURS_EVENT_D3");
                 d3_probe_valid = e ? 1 : 0; v = e ? atoi(e) : 0; }
    return (u64)(unsigned)v;
}

/* R171 SPURS_JC_EVENTS=mbox: lv2-accurate completion events for job-chain jobs.
 * A job tells the PPU it is done the way any SPU program does: it writes the outbound mailbox
 * (payload) and then the interrupt mailbox with code<<24 | data0, code < 64 = send_event(port),
 * 64..127 = throw_event(port).  lv2 delivers ONE event to the queue bound to that port:
 * source SYS_SPU_THREAD_EVENT_USER_KEY, data2 = port<<32 | data0, data3 = payload.
 * inFamous 2's Wwise chain (port 16 -> queue 5) counts exactly one event per submitted
 * "notify" job (image 301); the generic per-job broadcast below gave it ~5 events per
 * audio frame, desynchronised its outstanding-job count and, in bad phases, put the audio
 * thread into a 19k jobs/s resubmit storm that starved the command ring the main thread
 * blocks on (title/intro freezes).  Jobs that post nothing raise no event in this mode. */
static int jc_events_mbox(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("SPURS_JC_EVENTS"); v = (e && !strcmp(e, "mbox")) ? 1 : 0;
                 if (v) fprintf(stderr, "[R171] SPURS_JC_EVENTS=mbox: chain-job events from the job's own mailbox writes only%c", 10); }
    return v;
}
static void jc_signal_done(u32 jc_ea)
{
    if (jc_events_mbox()) {
        if (!g_spurs_job_mbox_valid) return;
        const u32 intr = g_spurs_job_mbox_intr, payload = g_spurs_job_mbox;
        g_spurs_job_mbox_valid = 0;
        const u32 code = intr >> 24;
        if (!intr || code >= 128u) return;              /* printf/other mailbox traffic: no event */
        const u32 port = code & 63u;
        const u32 q = spurs_port_queue(port);
        static unsigned long long n_ok = 0, n_noq = 0, n_fail = 0;
        if (!q) { if (n_noq++ < 8) fprintf(stderr, "[R171] chain 0x%08X job event to port %u: no queue bound%c", jc_ea, port, 10); return; }
        const int rc = sys_event_queue_push_by_id(q, 0xFFFFFFFF53505501ull, 0,
                                                  ((u64)port << 32) | (intr & 0x00FFFFFFu), payload);
        if (rc) { if (n_fail++ < 16) fprintf(stderr, "[R171] chain 0x%08X job event port %u -> queue %u FAILED rc=%d%c", jc_ea, port, q, rc, 10); }
        else if (n_ok++ < 4) fprintf(stderr, "[R171] chain 0x%08X job event port %u -> queue %u (%s, data0 0x%06X payload 0x%08X)%c",
                                     jc_ea, port, q, code < 64 ? "send" : "throw", intr & 0xFFFFFFu, payload, 10);
        return;
    }
    /* Carry the job's mailbox answer in the completion event. On hardware the
     * SPU's outbound/interrupt mailbox is what the SPURS event delivers; a
     * synthetic payload means the caller reads back zero for whatever it
     * asked. Keep the chain EA in data1 for anything that used it. */
    u64 d2 = 0, d3 = 0;
    (void)spurs_event_data3_probe();   /* prime d3_probe_valid */
    /* A PPU-side query submits a job and reads the reply out of the completion
     * event's data3 (the guest stores r7 at sp+0xB8 and reads the value back
     * from sp+0xBC, the low half). Tokyo Jungle asks each DSP module for its
     * input and output bus count this way -- commands 0x105 and 0x106 -- and
     * the SPU answers both in its outbound mailbox. We computed that mailbox
     * and then dropped it, pushing 0, so every module registered with a bus
     * count of 0 and the bus setup rejected the first bus it was asked for
     * ("sgxbus.c: out of range bus"): the check errors when count < requested,
     * and 0 < 1.
     *
     * Only the two count queries are forwarded. Neighbouring commands
     * (0x102/0x103/0x104) reply with LS ADDRESSES rather than counts, and
     * func_00201D38 stores each reply into module+0x1C/+0x20/+0x24. data2
     * stays 0 because the audio consumer uses it as a JOB INDEX: it waits until
     * data2 + 1 reaches the chain's job count, so a mailbox value there ends
     * the wait early. */
    enum { SGX_Q_IN_BUSES = 0x105, SGX_Q_OUT_BUSES = 0x106 };
    const int answers_a_count = (g_spurs_job_cmd == SGX_Q_IN_BUSES ||
                                 g_spurs_job_cmd == SGX_Q_OUT_BUSES);
    if (g_spurs_job_mbox_valid) {
        d2 = g_spurs_job_mbox;
        d3 = g_spurs_job_mbox_intr; (void)d3;
        g_spurs_job_mbox_valid = 0;
    }
    for (int i = 0; i < s_spurs_event_queue_n; i++) {
        int rc = sys_event_queue_push_by_id(s_spurs_event_queue[i],
                                            SPURS_EVENT_PORT, jc_ea, 0,
                                            d3_probe_valid
                                                ? spurs_event_data3_probe()
                                                : (answers_a_count ? d2 : 0));
        static int n = 0;
        if (n++ < 8)
            printf("[cellSpurs] chain 0x%08X work done -> event queue %u (rc=%d)\n",
                   jc_ea, s_spurs_event_queue[i], rc);
    }
}

/* R171 JOBWD: job-chain walker watchdog.  inFamous 2 intermittently freezes with the Ak audio
 * thread waiting on its completion queue while the walker host thread burns 98% CPU; these
 * globals plus a 2-second reporter say whether a job is stuck (which image/job) or the walker
 * itself is spinning (which chain position).  JOBWD=0 disables the reporter. */
extern volatile int g_jobwd_image, g_jobwd_active;
extern volatile uint32_t g_jobwd_job;
extern volatile uint64_t g_jobwd_fp, g_jobwd_start, g_jobwd_done;
extern uint64_t spu_jobwd_now_ms(void);
static volatile u32 s_jcwd_pc = 0, s_jcwd_jc = 0, s_jcwd_state = 0;   /* 1 walk, 2 job, 3 guard */
static volatile u64 s_jcwd_steps = 0, s_jcwd_guard_waits = 0, s_jcwd_exits = 0;
static DWORD WINAPI jcwd_thread(LPVOID p)
{
    (void)p;
    u64 last_steps = 0, last_done = 0, last_gw = 0; int reports = 0, tick = 0;
    extern volatile uint64_t g_jobwd_img_n[512], g_jobwd_img_us[512], g_jobwd_img_max[512];
    static u64 pn[512], pus[512];
    for (;;) {
        Sleep(2000);
        if (++tick % 5 == 0) {   /* [JOBSTAT] every 10 s: chain-job count and host time per image */
            char b[1024]; int p = 0; u64 tn = 0, tus = 0;
            for (int i = 0; i < 512; i++) {
                const u64 n = g_jobwd_img_n[i] - pn[i], us = g_jobwd_img_us[i] - pus[i];
                pn[i] = g_jobwd_img_n[i]; pus[i] = g_jobwd_img_us[i];
                if (!n) continue;
                tn += n; tus += us;
                if (p < (int)sizeof b - 80)
                    p += snprintf(b + p, sizeof b - p, " | img %d: %llu jobs %llu ms (avg %llu us, max %llu us)", i,
                                  (unsigned long long)n, (unsigned long long)(us / 1000), (unsigned long long)(us / n),
                                  (unsigned long long)g_jobwd_img_max[i]);
                g_jobwd_img_max[i] = 0;
            }
            fprintf(stderr, "[JOBSTAT] 10s: %llu chain jobs, %llu ms in jobs, walker steps %llu, guard waits %llu%s%c",
                    (unsigned long long)tn, (unsigned long long)(tus / 1000), (unsigned long long)s_jcwd_steps,
                    (unsigned long long)s_jcwd_guard_waits, p ? b : "", 10);
        }
        const u64 now = spu_jobwd_now_ms();
        const u64 steps = s_jcwd_steps, done = g_jobwd_done, gw = s_jcwd_guard_waits;
        const int job_stuck = g_jobwd_active && now - g_jobwd_start > 1500;
        const int walk_spin = (steps - last_steps) > 200000 && done == last_done;
        if ((job_stuck || walk_spin || (done == last_done && steps != last_steps)) && reports < 200) {
            reports++;
            fprintf(stderr, "[JOBWD] chain 0x%08X state %u pc 0x%08X cmd %016llX | 2s: steps %llu jobs %llu guard-waits %llu exits %llu"
                            " | job %s image %d job 0x%08X fp %016llX running %llu ms%c",
                    s_jcwd_jc, s_jcwd_state, s_jcwd_pc,
                    (unsigned long long)(s_jcwd_pc ? vm_read64(s_jcwd_pc) : 0),
                    (unsigned long long)(steps - last_steps), (unsigned long long)(done - last_done),
                    (unsigned long long)(gw - last_gw), (unsigned long long)s_jcwd_exits,
                    g_jobwd_active ? "ACTIVE" : "idle", g_jobwd_image, g_jobwd_job,
                    (unsigned long long)g_jobwd_fp,
                    (unsigned long long)(g_jobwd_active ? now - g_jobwd_start : 0), 10);
            fflush(stderr);
        }
        last_steps = steps; last_done = done; last_gw = gw;
    }
    return 0;
}
static void jcwd_start_once(void)
{
    static volatile LONG started = 0;
    const char* e = getenv("JOBWD");
    if (e && *e == '0') return;
    if (InterlockedCompareExchange(&started, 1, 0) == 0) {
        HANDLE th = CreateThread(NULL, 64u << 10, jcwd_thread, NULL, 0, NULL);
        if (th) CloseHandle(th);
        fprintf(stderr, "[JOBWD] job-chain watchdog on (JOBWD=0 off)%c", 10);
    }
}

/* [P172] JC_TRACE=<n>: per-lap job-chain trace (diagnostic, default off).  A lap starts at a
 * GUARD the walker passes; the line lists every command walked since the previous guard
 * (J=job desc:binary, S=sync, N=next, Z=zero word, oN=other op) plus a snapshot of the
 * guard's owner block (inFamous 2 Wwise layout: owner = guard - 0x80: +0 count, +4 write ptr,
 * +540 outstanding, +548/+558 buffer idx, +554 voice list (next +390, state +340, +39C)).
 * The first 24 laps, every 2000th lap and up to <n> laps with <= 1 job are printed. */
static int jc_trace_n(void)
{
    static int v = -2;
    if (v == -2) { const char* e = getenv("JC_TRACE"); v = e ? atoi(e) : 0;
                   if (v > 0) fprintf(stderr, "[P172] JC_TRACE=%d: per-lap job-chain trace%c", v, 10); }
    return v;
}
static int jc_trace_snap(char* b, int cap, u32 guard_ea)
{
    const u32 g = guard_ea - 0x80u;
    int p = snprintf(b, cap, " | own %08X: n %u wp %08X out %d i548 %u i558 %u v554 %08X", g,
                     vm_read32(g), vm_read32(g + 4), (int)vm_read32(g + 0x540), vm_read32(g + 0x548),
                     vm_read32(g + 0x558), vm_read32(g + 0x554));
    u32 v = vm_read32(g + 0x554);
    for (int k = 0; k < 6 && v && v < 0xF0000000u && p < cap - 64; k++) {
        p += snprintf(b + p, cap - p, " [%08X st %X 39C %X 394 %X]", v, vm_read32(v + 0x340),
                      vm_read32(v + 0x39C), vm_read32(v + 0x394));
        v = vm_read32(v + 0x390);
    }
    return p;
}

static void jc_execute(u32 entry_ea, u32 jc_ea, u32 size_desc)
{
    const int trace_n = jc_trace_n();
    char tl[1536]; int tp = 0, tjobs = 0; u64 tlap = 0; static int t_small = 0;
    tl[0] = 0;
    u32 pc = entry_ea, ret_pc = 0;
    jcwd_start_once(); s_jcwd_jc = jc_ea;
    int jobs = 0;
    /* Bound IDLE steps, not total steps. A SPURS job chain is frequently a
     * SERVICE LOOP -- guard, job, sync, next, repeat -- and is meant to run for
     * the lifetime of the subsystem. A flat cap on total commands therefore
     * kills a perfectly healthy chain: Tokyo Jungle's audio chain hit it after
     * 1332 jobs and its sound pipeline simply stopped. What actually indicates
     * a malformed chain is spinning through commands WITHOUT running a job, so
     * count that instead and reset it whenever real work happens. */
    int idle = 0;
    for (;; idle++) {
        if (idle > 4096) break;
        s_jcwd_pc = pc; s_jcwd_state = 1; s_jcwd_steps++;
        u64 cmd = vm_read64(pc);
        /* R171 JC_ZERO_WAIT_MS (default 0 = old behaviour): a zero command word is a slot the
         * PPU has not written YET (Wwise fills each double-buffered slot after the walker can
         * already be past the guard).  Skipping it as a NOP walks into the stale commands of an
         * earlier lap and loses jobs; wait for the word instead (bounded). */
        if (cmd == 0) {
            static int zw = -1;
            if (zw < 0) { const char* e = getenv("JC_ZERO_WAIT_MS"); zw = e ? atoi(e) : 0;
                          if (zw > 0) fprintf(stderr, "[R171] JC_ZERO_WAIT_MS=%d: chain walker waits for unwritten command words%c", zw, 10); }
            if (zw > 0) {
                static unsigned long long n_wait = 0, n_late = 0, n_timeout = 0;
                const uint64_t t0 = spu_jobwd_now_ms();
                n_wait++;
                while (cmd == 0 && spu_jobwd_now_ms() - t0 < (uint64_t)zw) {
                    s_jcwd_state = 4;
                    Sleep(0);
                    cmd = vm_read64(pc);
                }
                s_jcwd_state = 1;
                if (cmd) { if (n_late++ < 8) fprintf(stderr, "[R171] chain 0x%08X: command @0x%08X arrived after %llu ms (%016llX)%c",
                                                      jc_ea, pc, (unsigned long long)(spu_jobwd_now_ms() - t0), (unsigned long long)cmd, 10); }
                else if (n_timeout++ < 8) fprintf(stderr, "[R171] chain 0x%08X: command @0x%08X still zero after %d ms -- treated as NOP (%llu waits)%c",
                                                    jc_ea, pc, zw, n_wait, 10);
            }
        }
        u32 op  = (u32)(cmd & 7);
        u32 ext = (u32)(cmd & 127);
        if (trace_n > 0 && tp < (int)sizeof tl - 48) {
            if (cmd == 0) tp += snprintf(tl + tp, sizeof tl - tp, " Z@%X", pc & 0xFFFFu);
            else if (op == 0) { const u32 d = (u32)(cmd & ~7ull);
                tp += snprintf(tl + tp, sizeof tl - tp, " J%X:%X", d & 0xFFFFu, vm_read32(d + 4) & 0xFFFFF8u); }
            else if (op == 2) tp += snprintf(tl + tp, sizeof tl - tp, " S");
            else if (op == 3) tp += snprintf(tl + tp, sizeof tl - tp, " N>%X", (u32)cmd & 0xFFFFu);
            else if (!(op == 7 && ext == (7 | (1 << 3)))) tp += snprintf(tl + tp, sizeof tl - tp, " o%X", ext);
        }

        if (cmd != 0 && op == 0) {                    /* JOB */
            { extern u32 g_spurs_job_ls_handle; g_spurs_job_ls_handle = jc_ea; }
            s_jcwd_state = 2;
            jc_run_one_job((u32)(cmd & ~7ull), jobs++, size_desc);
            tjobs++;
            s_jcwd_state = 1;
            idle = 0;                    /* real work: the chain is healthy */
            /* Signal per JOB, not per lap. The application waits once for each
             * unit of work it queued -- this title notified the guard 7 times
             * and sat in event_queue_receive 30 times -- so batching the
             * completion into one event per pass leaves it waiting for
             * completions that, from its point of view, never arrive. */
            jc_signal_done(jc_ea);
            pc += 8; continue;
        }
        if (op == 1) { pc = (u32)(cmd & ~7ull); continue; }   /* RESET_PC */
        if (op == 3) { pc = (u32)(cmd & ~7ull); continue; }   /* NEXT     */
        if (op == 4) { ret_pc = pc + 8; pc = (u32)(cmd & ~7ull); continue; }  /* CALL */
        if (op == 7) {
            if (ext == (7 | (15 << 3))) {                     /* END */
                printf("[cellSpurs] chain 0x%08X: END after %d job(s)\n", jc_ea, jobs);
                if (jobs) jc_signal_done(jc_ea);
                return;
            }
            if (ext == (7 | (14 << 3))) {                     /* RET */
                if (!ret_pc) return;
                pc = ret_pc; ret_pc = 0; continue;
            }
            if (ext == (7 | (0 << 3))) {                      /* ABORT */
                printf("[cellSpurs] chain 0x%08X: ABORT\n", jc_ea);
                return;
            }
        }
        if (op == 7 && ext == (7 | (1 << 3))) {           /* GUARD */
            u32 g_ea = (u32)(cmd & ~127ull);
            s_jcwd_state = 3; s_jcwd_guard_waits++;
            if (!jg_wait(g_ea)) { s_jcwd_exits++; return; }   /* still closed */
            s_jcwd_state = 1;
            if (trace_n > 0) {
                const int show = tlap < 24 || tlap % 2000 == 0 || (tlap > 24 && tjobs <= 1 && t_small++ < trace_n);
                if (show) fprintf(stderr, "[P172] lap %llu t=%llu ms chain %08X jobs %d:%s%c", (unsigned long long)tlap,
                                  (unsigned long long)spu_jobwd_now_ms(), jc_ea, tjobs, tl, 10);
                tlap++; tjobs = 0;
                tp = snprintf(tl, sizeof tl, " G@%X", pc & 0xFFFFu);
                tp += jc_trace_snap(tl + tp, (int)sizeof tl - tp - 200, g_ea);
                tp += snprintf(tl + tp, sizeof tl - tp, " ||");
            }
            pc += 8; continue;
        }
        /* NOP(0) / SYNC+LWSYNC(2, one virtual SPU: nothing to wait for) /
         * FLUSH(5) / JOBLIST(6, nested arrays TODO) / SET_LABEL */
        pc += 8;
    }
    s_jcwd_exits++;
    printf("[cellSpurs] chain 0x%08X: 4096 commands with no job run after %d "
           "job(s) -- malformed?\n",
           jc_ea, jobs);
}

static DWORD WINAPI jc_thread(LPVOID p)
{
    int slot = (int)(intptr_t)p;
    /* TIMING PROBE (SPURS_JC_DELAY=ms): the real jm2 chain walker is async and
     * picks up jobs as the PPU appends them + fills their descriptors. Our walk
     * is one-shot; if it reads descriptors before the PPU populates the I/O
     * (n_dma=0, empty ioBuffer), deferring the walk should let real I/O appear.
     * Confirms timing-vs-never before committing to the async rewrite. */
    { const char* d = getenv("SPURS_JC_DELAY");
      if (d && *d) Sleep((unsigned)atoi(d)); }
    jc_execute(s_jobchains[slot].entry_ea, s_jobchains[slot].jc_ea,
               s_jobchains[slot].size_desc);
    s_jobchains[slot].running = 0;
    /* Tell the application the chain is done. Without this the walk finishes
     * in silence and a thread waiting on the attached queue never runs again. */
    jc_signal_done(s_jobchains[slot].jc_ea);
    return 0;
}

/* cellSpursRunJobChain -- start job chain execution (async, like the real one). */
/* cellSpursRunJobChain(const CellSpursJobChain* jobChain) -- ONE argument.
 *
 * This was declared (spurs, jobChain), so the ABI adapter fed it r3=spurs,
 * r4=<whatever>, and the chain it looked up was r4 -- never the handle
 * CreateJobChainWithAttribute registered. Every run logged
 * "UNKNOWN chain (no Create seen)" and returned CELL_OK without walking
 * anything, so the title sat in event_queue_receive waiting for SPU work that
 * was never started. Tokyo Jungle blocks its whole boot there.
 *
 * Verified against the guest: Create takes the chain in r4 (r3 is the CellSpurs)
 * and passes 0x02932A80; Run then arrives with r3=0x02932A80. Join and Shutdown
 * are the same one-argument shape. */
static s32 jc_start(u64 jc_ea, const char* who)
{
    static int s_off = -1;
    if (s_off < 0) s_off = getenv("PS3_NO_JOBCHAIN") ? 1 : 0;

    for (int i = 0; i < MAX_JOBCHAINS; i++) {
        if (s_jobchains[i].jc_ea != (u32)jc_ea) continue;
        s_jobchains[i].run_count++;
        if (s_jobchains[i].run_count <= 3) {
            printf("[cellSpurs] %s(jc=0x%08X) run#%d entry=0x%08X\n",
                   who, (u32)jc_ea, s_jobchains[i].run_count, s_jobchains[i].entry_ea);
            jc_dump_commands(who, s_jobchains[i].entry_ea, 16);
        }
        if (s_off || !s_jobchains[i].entry_ea) return CELL_OK;
        /* Coalesce: a chain already being walked must not start twice. */
        if (_InterlockedCompareExchange(&s_jobchains[i].running, 1, 0) == 0) {
            HANDLE th = CreateThread(NULL, 1u << 20, jc_thread, (LPVOID)(intptr_t)i, 0, NULL);
            if (th) CloseHandle(th);
            else s_jobchains[i].running = 0;
        }
        return CELL_OK;
    }
    printf("[cellSpurs] %s(jc=0x%08X) -- UNKNOWN chain (no Create seen)\n", who, (u32)jc_ea);
    return CELL_OK;
}

s32 cellSpursRunJobChain(u64 jc_ea)
{
    return jc_start(jc_ea, "RunJobChain");
}

/* cellSpursShutdownJobChain(CellSpursJobChain*) -- the other half of the pair.
 *
 * A title that starts a chain with Run/Kick ends it with Shutdown, then blocks
 * in Join until the chain has actually stopped. Leaving Shutdown unimplemented
 * is not harmless: the import logs UNIMPLEMENTED and returns whatever was in
 * r3, so the caller reads a garbage result for "did the shutdown take?" and a
 * title that checks it can decide the subsystem failed and tear itself down.
 *
 * Clearing `running` is what makes the pair honest: the host walker stops
 * grabbing new work, and the Join that follows has something true to observe.
 * Found in Tokyo Jungle, whose sound engine shuts its chain down during init
 * and then reports "failed to recv data from SPU" when the handshake does not
 * complete. */
s32 cellSpursShutdownJobChain(u64 jc_ea)
{
    static int _n = 0;
    if (_n++ < 8) printf("[cellSpurs] ShutdownJobChain(jc=0x%08X)\n", (u32)jc_ea);
    for (int i = 0; i < MAX_JOBCHAINS; i++) {
        if (s_jobchains[i].jc_ea == (u32)jc_ea) {
            s_jobchains[i].running = 0;   /* stop the walker; Join can now succeed */
            return CELL_OK;
        }
    }
    /* Not one of ours: still CELL_OK -- the chain is, trivially, not running. */
    return CELL_OK;
}

/* cellSpursJoinJobChain(const CellSpursJobChain*) -- one argument, as above. */
s32 cellSpursJoinJobChain(u64 jc_ea)
{
    static int _n = 0;
    if (_n++ < 8) printf("[cellSpurs] JoinJobChain(jc=0x%08X)\n", (u32)jc_ea);
    /* Run/Kick walk the chain on a host thread, so Join has to wait for it.
     * Returning at once told the title its jobs were done while they were
     * still running. The Simpsons Arcade Game runs and joins a chain every
     * frame, and those jobs DMA the fragment programs and vertices of draws
     * already queued in the FIFO: the RSX drain then reached the draws first
     * (black frames, glyph quads built from half-written vertices), and the
     * next frame's Run was coalesced away while this walker was still busy,
     * so that frame's data was never written at all. */
    for (int i = 0; i < MAX_JOBCHAINS; i++) {
        if (s_jobchains[i].jc_ea != (u32)jc_ea) continue;
        while (s_jobchains[i].running) {
#ifdef _WIN32
            Sleep(0);
#else
            usleep(0);
#endif
        }
    }
    return CELL_OK;
}

s32 cellSpursJobChainGetError(u64 jc_ea, u64 cause_out_ea)
{
    static int _n = 0;
    if (_n++ < 8) printf("[cellSpurs] JobChainGetError(jc=0x%08X)\n", (u32)jc_ea);
    if (cause_out_ea) vm_write32((u32)cause_out_ea, 0);
    return CELL_OK;
}

/* cellSpursKickJobChain(CellSpursJobChain* jobChain, u8 numReadyCount)
 *
 * Two arguments, jobChain in r3 -- the same one-handle shape as Run/Join, not
 * (spurs, jobChain). It is the *other* way a title starts a chain: Run for one
 * that was created ready to go, Kick to hand the SPUs another `numReadyCount`
 * units of an already-created chain. Titles built on the WWS job manager use
 * Kick, so leaving this a no-op meant their chains were created and then never
 * walked -- which looks exactly like a hang with no error anywhere. */
s32 cellSpursKickJobChain(u64 jc_ea, u32 numReadyCount)
{
    (void)numReadyCount;
    return jc_start(jc_ea, "KickJobChain");
}

/* =========================================================================
 * SPURS queues (PPU<->SPU bounded FIFO; LBP's audio instance uses these).
 * Log-only bring-up: capture the shapes before wiring real state.
 * =====================================================================*/

/* SDK ABI (cell/spurs/queue.h):
 *   _cellSpursQueueInitialize(CellSpurs*, CellSpursTaskset*, CellSpursQueue*,
 *       const void* buffer, u32 size, u32 depth, CellSpursQueueDirection) */
s32 _cellSpursQueueInitialize(u64 spurs_ea, u64 taskset_ea, u64 queue_ea,
                              u64 buffer_ea, u32 size, u32 depth, u32 direction)
{
    (void)spurs_ea;
    if (!queue_ea || !buffer_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    if (!size || !depth)         return CELL_SPURS_TASK_ERROR_INVAL;

    /* Same 128-byte big-endian line as the LF variant below, and the title's
     * own call says so: it passes q=0x032B2980 with buffer=0x032B2A00, exactly
     * 0x80 apart, just as its LFQueue passes 0x4059FD00/0x4059FD80. The
     * consumer is task 5 (image 1), whose CellSpursTaskArgument points into
     * this same 0x032B2xxx block and which reads the line with GETLLAR/PUTLLC,
     * so leaving it uninitialised handed recompiled SPU code a zero-depth
     * queue to reason about. */
    uint32_t q = (uint32_t)queue_ea;
    for (uint32_t o = 0; o < 128; o += 4) vm_write32(q + o, 0);
    /* The first 16 bytes are push1's live reservation/completion state.  In
     * particular +0x0E is m_h8; writing depth at +0x0C makes a fresh queue look
     * completely full (m_h8 == depth).  Ring metadata starts at +0x10. */
    vm_write32(q + 0x10, size);
    vm_write32(q + 0x14, depth);
    vm_write64(q + 0x18, (u64)(uint32_t)buffer_ea);
    /* NOT direction at +0x24 and NOT init at +0x2C. Both sit inside the
     * 16-byte group at +0x20..+0x2F that the consumer owns and shifts wholesale
     * (shlqbyi <group>,1 at 0x12A0C). The trace shows the two values we used to
     * write there marching through it one byte per dequeue --
     *   +0x24: 00000002 -> 00000200 -> 00020000 -> 02000000
     *   +0x2C: 00000001 -> 00000100 -> 00010000 -> 01000000
     * -- i.e. we were feeding garbage into the SPU's own state every cycle.
     * Whatever holds direction/init, it is not these offsets. */
    vm_write64(q + 0x70, (u64)(uint32_t)taskset_ea);
    memset(vm_base + (uint32_t)buffer_ea, 0, (size_t)size * depth);

    static int _n = 0;
    if (_n++ < 8)
        printf("[cellSpurs] _QueueInitialize(taskset=0x%08X q=0x%08X buf=0x%08X size=%u depth=%u dir=%u) -> BE line written\n",
               (u32)taskset_ea, (u32)queue_ea, (u32)buffer_ea, size, depth, direction);
    return CELL_OK;
}

s32 cellSpursQueueClear(u64 queue_ea)
{
    static int _n = 0;
    if (_n++ < 8) printf("[cellSpurs] QueueClear(q=0x%08X)\n", (u32)queue_ea);
    return CELL_OK;
}

/* SDK ABI: cellSpursQueuePushBody(CellSpursQueue*, const void* buffer, bool isBlocking) */
s32 cellSpursQueuePushBody(u64 queue_ea, u64 data_ea, u32 isBlocking)
{
    /* SPURS_QUEUE_PUSH=1 -- EXPERIMENT, off by default.
     *
     * A deliberately self-revealing probe rather than a finished push. The
     * waiter half of the protocol is known (see PROGRESS.md phase 19): the
     * 16-byte group at queue+0x20 is a ring of blocked task ids, byte[0] the
     * count and byte[13] the cursor. The DATA half is not: which ring slot an
     * element belongs in has never been observed, because nothing has ever
     * pushed and so the consumer has never taken its non-empty path.
     *
     * So push into a CANDIDATE slot (a counter kept in the push1 half of the
     * line at +0x0C), wake the registered waiter, and let the consumer's own
     * DMA say where it actually looks. Its read address names the correct slot
     * whether or not the guess was right, which turns one run into the answer.
     * Everything is logged for that reason. */
    static int s_exp = -1;
    if (s_exp < 0) s_exp = getenv("SPURS_QUEUE_PUSH") ? 1 : 0;

    static int _n = 0;
    if (!s_exp) {
        if (_n < 8)
            printf("[cellSpurs] QueuePushBody(q=0x%08X data=0x%08X blocking=%u)\n",
                   (u32)queue_ea, (u32)data_ea, isBlocking);
        else if (_n == 8)
            printf("[cellSpurs] QueuePushBody further logs suppressed\n");
        _n++;
        return CELL_OK;
    }

    if (!queue_ea || !data_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    uint32_t q   = (uint32_t)queue_ea;
    uint32_t sz  = vm_read32(q + 0x10);
    uint32_t dep = vm_read32(q + 0x14);
    uint32_t buf = (uint32_t)vm_read64(q + 0x18);
    uint32_t tsp = (uint32_t)vm_read64(q + 0x70);      /* eaSignal: taskset */
    if (!sz || !dep || !buf) return CELL_SPURS_TASK_ERROR_INVAL;

    /* Derived ring model (see PROGRESS.md phase 22). Working the dataflow of
     * the consumer's decision at 0x12914..0x129D0 backwards:
     *
     *   r11 = normalise(W0)        W0 = queue+0x00
     *   r2  = normalise(W1)        W1 = queue+0x04
     *   r60 = W3                   W3 = queue+0x0C
     *     where normalise(v) is  cgti/nor/selb  ==  (v >= 0 ? v : ~v)
     *
     *   r81 = r2 - r11
     *   r79 = (r2 + r60) - (r11 - r60)  =  r2 - r11 + 2*r60
     *   r68 = (r11 > r2) ? r79 : r81    =  (r2 - r11) mod 2*r60
     *   ceqi r77, r68, 0                -> the EMPTY test
     *
     * and, separately at 0x129A8..0x129B8,
     *
     *   r14 = (r60 > r11) ? r11 : r11 - r60   =  index mod r60
     *
     * which is the buffer slot. So this is an ordinary ring: W0 is the pop
     * index, W1 the push index, indices run modulo 2*W3, and the slot is the
     * index modulo W3 -- the standard scheme that keeps "full" distinguishable
     * from "empty". W3 is therefore the depth.
     *
     * And that is the bug: the initialiser left queue+0x0C at ZERO, so r60 = 0
     * and the whole computation degenerates -- occupancy is (W1-W0) mod 0 and
     * the slot is index mod 0. With both indices at 0 the empty test is
     * trivially true, which is precisely why the consumer parked every time and
     * never once read the element buffer, whatever the producer wrote. */
    uint8_t* qp = vm_base + q;

    int32_t  sync  = (int32_t)vm_read32(q + 0x00);          /* W0, pop  */
    int32_t  tail  = (int32_t)vm_read32(q + 0x04);          /* W1, push */
    int32_t  mod   = (int32_t)vm_read32(q + 0x0C);          /* W3       */
    if (mod <= 0) mod = (int32_t)dep;                       /* pre-fix queues */

    int32_t  nsync = (sync < 0) ? ~sync : sync;             /* normalise */
    int32_t  cur   = (tail < 0) ? ~tail : tail;
    uint32_t slot  = (uint32_t)(cur % mod);

    memcpy(vm_base + buf + (size_t)slot * sz, vm_base + (uint32_t)data_ea, sz);

    int32_t  nxt   = (cur + 1) % (2 * mod);
    int32_t  ntail = nxt;
    vm_write32(q + 0x04, (uint32_t)ntail);

    /* Waiter ring at +0x20: byte[0] = count, byte[1..12] = task ids. */
    uint32_t waiters = qp[0x20];
    int woke = -1;
    if (waiters > 0 && waiters <= 3 && tsp) {
        /* The entry area is m_bs[1..3] ONLY -- bytes 0x21..0x23. Shifting the
         * whole 16-byte group (as a first cut of this did) walks straight over
         * direction at +0x24 and init at +0x2C and corrupts the header; the
         * trace caught it as direction turning from 00000002 into 00020000. */
        woke = qp[0x21];
        qp[0x21] = qp[0x22];
        qp[0x22] = qp[0x23];
        qp[0x23] = 0;
        qp[0x20] = (uint8_t)(waiters - 1);
        extern void spu_taskset_signal_task(uint32_t taskset_ea, uint32_t taskId);
        spu_taskset_signal_task(tsp, (uint32_t)woke);
    }

    if (_n++ < 16)
        printf("[cellSpurs] QueuePush#%d q=0x%08X slot=%u (cur=%u->%u) buf=0x%08X+0x%X "
               "size=%u depth=%u waiters=%u woke=%d sync=%d/%d tail=%d->%d taskset=0x%08X\n",
               _n, q, slot, cur, nxt, buf, slot * sz, sz, dep, waiters, woke, sync, nsync, tail, ntail, tsp);
    return CELL_OK;
}

/* =========================================================================
 * Misc SPURS surface the title links (logged CELL_OK stubs).
 * =====================================================================*/

s32 cellSpursRequestIdleSpu(u64 spurs_ea)
{
    static int _n = 0;
    if (_n++ < 4) printf("[cellSpurs] RequestIdleSpu(spurs=0x%08X)\n", (u32)spurs_ea);
    return CELL_OK;
}

s32 cellSpursGetInfo(u64 spurs_ea, u64 info_ea)
{
    struct SpursInst* si = spurs_inst_find((u32)spurs_ea);
    static int _n = 0;
    if (_n++ < 4) printf("[cellSpurs] GetInfo(spurs=0x%08X info=0x%08X)\n", (u32)spurs_ea, (u32)info_ea);
    if (info_ea) {
        vm_write32((u32)info_ea, si ? si->nspus : 1);   /* nSpus */
        for (int o = 4; o < 0x28; o += 4) vm_write32((u32)info_ea + o, 0);
    }
    return CELL_OK;
}

s32 cellSpursSetExceptionEventHandler(u64 spurs_ea, u64 handler_ea, u64 arg_ea)
{
    static int _n = 0;
    if (_n++ < 4)
        printf("[cellSpurs] SetExceptionEventHandler(spurs=0x%08X handler=0x%08X)\n",
               (u32)spurs_ea, (u32)handler_ea);
    return CELL_OK;
}

s32 cellSpursGetWorkloadInfo(u64 spurs_ea, u32 wid, u64 info_ea)
{
    static int _n = 0;
    if (_n++ < 8) printf("[cellSpurs] GetWorkloadInfo(wid=%u info=0x%08X)\n", wid, (u32)info_ea);
    /* Log every CHANGE of (wid, readyCount, contention): a guest that polls
     * this until a workload drains (inFamous func_0041DC00 waits for
     * readyCount==0 && contention==0) shows here what it is stuck on. */
    if (spurs_ea && wid < CELL_SPURS_MAX_WORKLOAD) {
        static u32 s_last = 0xFFFFFFFFu; static int s_cn = 0;
        u32 se0 = (u32)spurs_ea;
        u32 key = (wid << 16) | ((u32)vm_base[se0 + SPURS_WKL_READY1 + wid] << 8) | vm_base[se0 + SPURS_WKL_CURCONT + wid];
        if (key != s_last && s_cn++ < 200) {
            s_last = key;
            fprintf(stderr, "[cellSpurs] GetWorkloadInfo wid=%u readyCount=%u contention=%u\n",
                    wid, (key >> 8) & 0xFF, key & 0xFF);
        }
    }

    if (!spurs_ea || !info_ea)
        return CELL_SPURS_CORE_ERROR_NULL_POINTER;
    if (wid >= CELL_SPURS_MAX_WORKLOAD || !s_workloads[wid].in_use)
        return CELL_SPURS_CORE_ERROR_INVAL;

    /* CellSpursWorkloadInfo is a GUEST out-buffer (cell/spurs/workload_types.h),
     * big-endian, 32-bit pointers. This used to return CELL_OK writing NOTHING,
     * so the guest read whatever stale bytes sat at info_ea as the descriptor.
     * Fill the header from our workload mirror + the live BE instance counters;
     * zero the pointer/name/hook fields we do not track. Byte fields are single
     * bytes (endian-neutral); multi-byte fields go through vm_write* (BE). */
    const SpursWorkload* w = &s_workloads[wid];
    u32 base = (u32)info_ea;
    for (u32 o = 0; o < 0x30; o += 4) vm_write32(base + o, 0);   /* header clean */

    vm_write64(base + 0x00, w->data);                            /* data        */
    for (int b = 0; b < 8 && b < CELL_SPURS_MAX_SPU; b++)
        *(vm_base + base + 0x08 + b) = w->priority[b];           /* priority[8] */
    vm_write32(base + 0x10, (u32)(uintptr_t)w->pm);              /* policyModule (32-bit EA) */
    vm_write32(base + 0x14, w->sizePm);                          /* sizePolicyModule */
    /* nameClass/nameInstance (0x18/0x1C): not tracked -> left 0 by the zero above. */

    u32 se = (u32)spurs_ea;
    *(vm_base + base + 0x20) = *(vm_base + se + SPURS_WKL_CURCONT + wid);  /* contention   */
    *(vm_base + base + 0x21) = (u8)w->minContention;                       /* minContention */
    *(vm_base + base + 0x22) = (u8)w->maxContention;                       /* maxContention */
    *(vm_base + base + 0x23) = *(vm_base + se + SPURS_WKL_READY1 + wid);   /* readyCount   */
    *(vm_base + base + 0x24) = *(vm_base + se + SPURS_WKL_IDLE2  + wid);   /* idleSpuRequest */
    u32 sig = vm_read32(se + SPURS_WKL_SIGNAL1) >> 16;
    *(vm_base + base + 0x25) = (sig & (0x8000u >> wid)) ? 1 : 0;           /* hasSignal    */
    return CELL_OK;
}

s32 cellSpursShutdownWorkload(u64 spurs_ea, u32 wid)
{
    static int _n = 0;
    if (_n++ < 64) printf("[cellSpurs] ShutdownWorkload(wid=%u)\n", wid);
    if (!spurs_wkl_lifecycle_on()) return CELL_OK;   /* v101 */
    return spurs_wkl_shutdown_impl((u32)spurs_ea, wid, "");
}

s32 cellSpursWaitForWorkloadShutdown(u64 spurs_ea, u32 wid)
{
    static int _n = 0;
    if (_n++ < 64) printf("[cellSpurs] WaitForWorkloadShutdown(wid=%u)\n", wid);
    if (!spurs_wkl_lifecycle_on()) return CELL_OK;   /* v101 */
    return spurs_wkl_wait_impl((u32)spurs_ea, wid, "");
}

/* =========================================================================
 * Task LS patterns and context save area sizing.
 *
 * Saints Row 2 builds every taskset through this sequence (func_009F56E0):
 *
 *   _cellSpursTasksetAttributeInitialize(attr, ...)
 *   cellSpursTasksetAttributeSetName(attr, name)
 *   cellSpursTasksetAttributeSetTasksetSize(attr, size)
 *   cellSpursCreateTasksetWithAttribute(spurs, taskset, attr)
 *   ...
 *   _cellSpursQueueInitialize(...) / cellSpursQueueAttachLv2EventQueue(q)
 *   cellSpursTaskGetReadOnlyAreaPattern(&ro,   elf)
 *   cellSpursTaskGetContextSaveAreaSize (&size, &ls)   ls = default & ~ro
 *   _cellSpursTaskAttributeInitialize(attr2, .., size, &ls, ..)
 *
 * The ABI of the last two is not guesswork -- it is readable straight off the
 * call site. At 0x009F5878 the caller sets r3 = sp+0x70 and r4 = r25, where
 * r25 = sp+0x90 (set at 0x009F5744) is exactly the 16-byte pattern the two
 * `andc`+`std` pairs at 0x009F5888..0x009F5894 had just built. After the call
 * it does `lwz r11, 0x70(r1)` -- reading a u32 back out of r3 -- checks it
 * against 0x2D400 and bails if it does not fit, then stores the size and the
 * pattern pointer side by side at sp+0x84/sp+0x88 for the task attribute. So:
 *
 *   s32 cellSpursTaskGetContextSaveAreaSize(u32* size_out,
 *                                           const CellSpursTaskLsPattern* ls);
 *
 * A task's context save area holds the SPU register file plus every 2 KB local
 * store block the task may dirty; the read-only blocks (its code/rodata, which
 * can simply be reloaded from the ELF) are masked out by the caller first.
 * Local store is 256 KB and the pattern is 128 bits, so one bit == 2 KB.
 * =====================================================================*/

#define SPURS_LS_BLOCK      2048u    /* 256 KB local store / 128 pattern bits */
#define SPURS_CTX_HDR       2048u    /* 128 SPU registers x 16 bytes          */
#define SPURS_CTX_ALIGN     128u     /* CELL_SPURS_TASK_CONTEXT_SAVE_AREA_ALIGN */

/* cellSpursTaskGetReadOnlyAreaPattern(CellSpursTaskLsPattern* pattern,
 *                                     const void* elf)
 *
 * Mark the local-store blocks covered by the image's NON-writable PT_LOAD
 * segments. The caller clears these out of its default pattern, so anything we
 * fail to report simply ends up saved as well -- costing context space, never
 * correctness. Bit numbering is PPC-style (bit 0 = MSB of the first word), and
 * the size path below only ever popcounts the result, so a bit-order mistake
 * here cannot change the computed size.
 */
s32 cellSpursTaskGetReadOnlyAreaPattern(u64 pattern_ea, u64 elf_ea)
{
    if (!pattern_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    u64 word[2] = { 0, 0 };

    if (elf_ea) {
        const uint8_t* img = vm_base + (uint32_t)elf_ea;
        if (img[0] == 0x7F && img[1] == 'E' && img[2] == 'L' && img[3] == 'F' &&
            img[4] == 1 /*ELFCLASS32*/ && img[5] == 2 /*ELFDATA2MSB*/) {
            uint32_t e_phoff = ((uint32_t)img[0x1C] << 24) | ((uint32_t)img[0x1D] << 16) |
                               ((uint32_t)img[0x1E] << 8)  |  (uint32_t)img[0x1F];
            uint16_t e_phentsize = (uint16_t)((img[0x2A] << 8) | img[0x2B]);
            uint16_t e_phnum     = (uint16_t)((img[0x2C] << 8) | img[0x2D]);
            if (e_phentsize < 0x20) e_phentsize = 0x20;

            for (uint16_t i = 0; i < e_phnum; i++) {
                const uint8_t* ph = img + e_phoff + (size_t)i * e_phentsize;
                #define BE32(p) (((uint32_t)(p)[0] << 24) | ((uint32_t)(p)[1] << 16) | \
                                 ((uint32_t)(p)[2] << 8)  |  (uint32_t)(p)[3])
                uint32_t p_type  = BE32(ph + 0x00);
                uint32_t p_vaddr = BE32(ph + 0x08);
                uint32_t p_memsz = BE32(ph + 0x14);
                uint32_t p_flags = BE32(ph + 0x18);
                #undef BE32
                if (p_type != 1 /*PT_LOAD*/ || !p_memsz) continue;
                if (p_flags & 2 /*PF_W*/)                continue;   /* writable: must be saved */
                if ((uint64_t)p_vaddr + p_memsz > SPU_LS_SIZE) continue;

                uint32_t first = p_vaddr / SPURS_LS_BLOCK;
                uint32_t last  = (p_vaddr + p_memsz - 1) / SPURS_LS_BLOCK;
                for (uint32_t b = first; b <= last && b < 128; b++)
                    word[b >> 6] |= (u64)1 << (63 - (b & 63));
            }
        }
    }

    vm_write64((uint32_t)pattern_ea,     word[0]);
    vm_write64((uint32_t)pattern_ea + 8, word[1]);

    static int _n = 0;
    if (_n++ < 8)
        printf("[cellSpurs] TaskGetReadOnlyAreaPattern(elf=0x%08X) -> %016llX %016llX\n",
               (u32)elf_ea, (unsigned long long)word[0], (unsigned long long)word[1]);
    return CELL_OK;
}

/* cellSpursTaskGetContextSaveAreaSize(u32* size_out,
 *                                     const CellSpursTaskLsPattern* lsPattern) */
s32 cellSpursTaskGetContextSaveAreaSize(u64 size_out_ea, u64 pattern_ea)
{
    if (!size_out_ea || !pattern_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    u64 hi = vm_read64((uint32_t)pattern_ea);
    u64 lo = vm_read64((uint32_t)pattern_ea + 8);

    uint32_t blocks = 0;
    for (int i = 0; i < 64; i++) {
        if (hi & ((u64)1 << i)) blocks++;
        if (lo & ((u64)1 << i)) blocks++;
    }

    uint32_t size = SPURS_CTX_HDR + blocks * SPURS_LS_BLOCK;
    size = (size + (SPURS_CTX_ALIGN - 1)) & ~(SPURS_CTX_ALIGN - 1);

    vm_write32((uint32_t)size_out_ea, size);

    static int _n = 0;
    if (_n++ < 8)
        printf("[cellSpurs] TaskGetContextSaveAreaSize(ls=%016llX %016llX, %u blocks) -> %u (0x%X)\n",
               (unsigned long long)hi, (unsigned long long)lo, blocks, size, size);
    return CELL_OK;
}

/* cellSpursTasksetAttributeSetTasksetSize(CellSpursTasksetAttribute*, u32 size)
 * CreateTaskset ignores the attribute entirely (it builds the real BE layout
 * itself), so record the request and accept it. */
s32 cellSpursTasksetAttributeSetTasksetSize(CellSpursTasksetAttribute* attr, u32 size)
{
    if (!attr) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    static int _n = 0;
    if (_n++ < 8) printf("[cellSpurs] TasksetAttributeSetTasksetSize(size=%u)\n", size);
    return CELL_OK;
}

/* cellSpursQueueAttachLv2EventQueue(CellSpursQueue*)
 * The queue's completion signalling runs through our own event-flag path, so
 * there is no lv2 queue to bind; accept and log. */
s32 cellSpursQueueAttachLv2EventQueue(u64 queue_ea)
{
    static int _n = 0;
    if (_n++ < 8) printf("[cellSpurs] QueueAttachLv2EventQueue(q=0x%08X)\n", (u32)queue_ea);
    return CELL_OK;
}

/* _cellSpursLFQueueInitialize(void* pTasksetOrSpurs, CellSpursLFQueue* pQueue,
 *      const void* buffer, u32 size, u32 depth, u32 direction)
 *
 * This one CANNOT be a stub, because the consumer is recompiled SPU code that
 * reads the queue out of main memory itself. Disassembling image 1 around the
 * task's WAIT_SIGNAL call site (LS 0x12B80) shows the real protocol: it builds
 * a 128-byte line at LS 0x80 and commits it with
 *
 *   wrch MFC_LSA,0x80 / MFC_EAH / MFC_EAL / MFC_Size,128 / MFC_Cmd,0xB4 (PUTLLC)
 *   rdch MFC_RdAtomicStat ; brnz -> retry
 *
 * i.e. a GETLLAR/PUTLLC lock-line atomic on the queue's own cache line. So the
 * bytes in guest memory ARE the interface, and they have to be the real
 * big-endian CellSyncLFQueue: one 128-byte, 128-aligned line.
 *
 *   0x00 pop1     0x10 size     0x18 buffer(u64)  0x24 direction  0x2C init
 *   0x08 push1    0x14 depth    0x20 bs[4]        0x28 v1         0x70 eaSignal
 *
 * The title's own call corroborates the size: it passes q=0x4059FD00 with
 * buffer=0x4059FD80, exactly 128 bytes later.
 *
 * NOTE: libs/sync/cellSync.c has a CellSyncLFQueue too, but that one is a
 * HOST-native struct (atomic_uint, a 64-bit host buffer pointer). It is fine
 * for a queue both of whose ends are HLE, and completely wrong here -- writing
 * it into guest memory would hand the SPU a host pointer where a 32-bit big-
 * endian EA belongs. Hence a separate, guest-accurate initializer rather than
 * delegating to it.
 *
 * What is set here is only what the arguments determine outright: size, depth,
 * buffer, direction, and the init flag, over a zeroed line (the documented
 * empty state). The bs[]/v1 slot state machine is left zero -- see PROGRESS.md;
 * it is not guessed at. */
s32 _cellSpursLFQueueInitialize(u64 owner_ea, u64 queue_ea, u64 buffer_ea,
                                u32 size, u32 depth, u32 direction)
{
    if (!queue_ea || !buffer_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;
    if (!size || !depth)         return CELL_SPURS_TASK_ERROR_INVAL;

    uint32_t q = (uint32_t)queue_ea;
    for (uint32_t o = 0; o < 128; o += 4) vm_write32(q + o, 0);

    /* The first 16 bytes are push1's live reservation/completion state.  In
     * particular +0x0E is m_h8; writing depth at +0x0C makes a fresh queue look
     * completely full (m_h8 == depth). Ring metadata starts at +0x10. */
    vm_write32(q + 0x10, size);
    vm_write32(q + 0x14, depth);
    vm_write64(q + 0x18, (u64)(uint32_t)buffer_ea);   /* bcptr<void,u64> */
    /* CellSyncLFQueue's public ABI places m_bs[4] at +0x20, direction at
     * +0x24, m_v1 at +0x28, and init at +0x2C. The empty non-ANY2ANY state has
     * all four m_bs bytes set to 0xFF and init released back to zero. */
    vm_write32(q + 0x20, 0xFFFFFFFFu);
    vm_write32(q + 0x24, direction);
    vm_write32(q + 0x28, 0);
    vm_write32(q + 0x2C, 0);
    vm_write64(q + 0x70, (u64)(uint32_t)owner_ea);    /* eaSignal <- taskset/spurs */

    memset(vm_base + (uint32_t)buffer_ea, 0, (size_t)size * depth);

    static int _n = 0;
    if (_n++ < 8)
        printf("[cellSpurs] _LFQueueInitialize(owner=0x%08X q=0x%08X buf=0x%08X size=%u depth=%u dir=%u) -> BE line written\n",
               (u32)owner_ea, (u32)queue_ea, (u32)buffer_ea, size, depth, direction);
    return CELL_OK;
}

/* _cellSpursLFQueuePushBody(CellSpursLFQueue*, const void*, u32 isBlocking)
 *
 * Implement the single-producer PPU side of the CellSyncLFQueue protocol used
 * by inFamous' FIOS reader. push1.m_h8 reserves a position and push1.m_h5
 * publishes it after the payload copy. Indices run modulo 2*depth; the SPU
 * consumes the same big-endian cache line with GETLLAR/PUTLLC. */

/* ---- v117 FIOSREQ=1: capture the FIOS SPU task's request format (fios_req.txt). For the first 40 LFQueue
 * pushes: the pushed record (64 bytes), then every DMA the async task thread issues until the next push,
 * with the first 32 bytes of its source (GET: main memory; PUT: local store). Goal: replace the task's
 * ~1.8 ms emulated psarc block decompress with a native zlib inflate. */
static FILE* s_rq; static int s_rq_n = -1; static volatile long s_rq_dma;
static int fiosreq_on(void)
{
    if (s_rq_n < 0) { const char* e = getenv("FIOSREQ"); s_rq_n = 0;
        if (e && *e == '1') { s_rq = fopen("fios_req.txt", "w");
            if (s_rq) fprintf(stderr, "[FIOSREQ] v117 capturing the first 40 FIOS requests -> fios_req.txt\n"); } }
    return s_rq != NULL && s_rq_n <= 40;
}
static void fiosreq_hex(const uint8_t* p, unsigned n)
{
    for (unsigned i = 0; i < n; i++) fprintf(s_rq, "%02X%s", p[i], (i & 3) == 3 ? " " : "");
}
void ps3_fiosreq_push(uint32_t q, uint32_t data, uint32_t size)
{
    if (!fiosreq_on()) return;
    s_rq_n++; s_rq_dma = 0;
    fprintf(s_rq, "\nPUSH #%d q=%08X data=%08X size=%u:\n  ", s_rq_n, q, data, size);
    fiosreq_hex(vm_base + data, size < 64 ? size : 64);
    fprintf(s_rq, "\n");
    {   /* v118: the status record (word 4) and the event flag (word 5) lines before the task runs */
        uint32_t st = vm_read32(data + 16) & ~127u, ef = vm_read32(data + 20) & ~127u;
        fprintf(s_rq, "  BEFORE status line %08X: ", st); fiosreq_hex(vm_base + st, 128);
        fprintf(s_rq, "\n  BEFORE flag line %08X: ", ef); fiosreq_hex(vm_base + ef, 128); fprintf(s_rq, "\n"); }
    fflush(s_rq);
}
/* v118: after the PPU wakes from its wait on the FIOS flag: the same two lines again */
void ps3_fiosreq_woke(uint32_t flag, u16 bits)
{
    if (!fiosreq_on() || !s_rq_n) return;
    uint32_t ef = flag & ~127u, st = 0x405C4980u;
    fprintf(s_rq, "  WOKE flag=%08X bits=%04X\n  AFTER status line %08X: ", flag, bits, st); fiosreq_hex(vm_base + st, 128);
    fprintf(s_rq, "\n  AFTER flag line %08X: ", ef); fiosreq_hex(vm_base + ef, 128); fprintf(s_rq, "\n"); fflush(s_rq);
}
void ps3_fiosreq_dma(uint32_t cmd, uint32_t lsa, uint32_t ea, uint32_t size, const uint8_t* ls)
{
    if (!fiosreq_on() || !s_rq_n || s_rq_dma++ > 60) return;
    const int put = (cmd & 0x20) && !(cmd & 0x40);
    fprintf(s_rq, "  DMA %s cmd=%02X lsa=%05X ea=%08X size=%u src:", put ? "PUT" : "GET", cmd, lsa, ea, size);
    const uint8_t* src = put ? ls + (lsa & 0x3FFFF) : vm_base + ea;
    fiosreq_hex(src, size < 32 ? size : 32);
    fprintf(s_rq, "\n"); fflush(s_rq);
}

/* ---- v119 FIOS_HLE=1: native zlib for the FIOS SPU decompress task (inFamous world streaming).
 * play115/116 FTL: every request = ~1.8 ms of emulated SPU decompression, ~180 per streaming burst = the
 * world-movement hitch (play114: slower streaming -> longer hitches). Request (LFQueue 0x405C4780 entry,
 * 32 B BE, from play117/118 captures): w0 src (zlib block, any alignment), w1 dst, w2 compressed size,
 * w3 bytes to write, w4 status record (busy word at w4 & ~3: 1 -> 0 when done), w5 event flag EA,
 * w6 = (EF bit << 16) | offset into the decompressed 64 KB block. Done = busy word 0, then EF bit set.
 * Anything not recognised (not 78 xx zlib, inflate fails/short, sizes out of range) goes to the SPU task
 * exactly as before. FIOS_HLE_Q=<hex> overrides the queue (default 405C4780). */
extern int stbi_zlib_decode_noheader_buffer(char* obuffer, int olen, const char* ibuffer, int ilen);   /* stb_image (cellPngDec.c) */
static volatile long long s_fh_n, s_fh_us, s_fh_fb, s_fh_t0;
static int fios_hle_try(uint32_t q, uint32_t data)
{
    static int on = -1; static uint32_t hq = 0x405C4780u;
    if (on < 0) { const char* e = getenv("FIOS_HLE"); on = (e && *e == '1') ? 1 : 0;
                  const char* e2 = getenv("FIOS_HLE_Q"); if (e2) hq = (uint32_t)strtoul(e2, 0, 16);
                  if (on) fprintf(stderr, "[FIOS-HLE] v119 native zlib for FIOS decompress requests on queue 0x%08X (FIOS_HLE=0 = off)\n", hq); }
    if (!on || q != hq) return 0;
    uint32_t w[8]; for (int i = 0; i < 8; i++) w[i] = vm_read32(data + 4u * (uint32_t)i);
    const uint32_t src = w[0], dst = w[1], csz = w[2], n = w[3], st = w[4], ef = w[5];
    const uint32_t bits = w[6] >> 16, skip = w[6] & 0xFFFFu;
    LARGE_INTEGER t0, t1, f; QueryPerformanceCounter(&t0);
    const char* why = 0;
    if (!bits || !src || !dst || !csz || !n || !st || !ef) why = "zero field";
    else if (csz > 0x20000u || skip + n > 0x10000u) why = "size out of range";
    /* v119b: w0 points PAST the 2-byte zlib header (play119: every block started ED 9D / ED 7D; the task DMAs
     * from src-2, where 78 DA sits) -> raw deflate at src, w2 bytes long. */
    else if (src < 2 || vm_base[src - 2] != 0x78) why = "no zlib header before the data";
    if (!why) {
        static SPU_PM_TLS char* t_buf = 0;
        if (!t_buf) t_buf = (char*)malloc(0x10000);
        if (!t_buf) why = "no buffer";
        else {
            int got = stbi_zlib_decode_noheader_buffer(t_buf, 0x10000, (const char*)(vm_base + src), (int)csz);
            if (got < (int)(skip + n)) why = got < 0 ? "inflate failed" : "inflate short";
            else {
                memcpy(vm_base + dst, t_buf + skip, n);
                atomic_thread_fence(memory_order_seq_cst);
                vm_write32(st & ~3u, 0);                   /* busy word 1 -> 0 */
                spurs_ef_set_from_spu(ef, (u16)bits);      /* wake the mediathread */
            }
        }
    }
    QueryPerformanceCounter(&t1); QueryPerformanceFrequency(&f);
    const long long us = (t1.QuadPart - t0.QuadPart) * 1000000 / f.QuadPart;
    if (why) {
        InterlockedIncrement64(&s_fh_fb);
        static int nf = 0;
        if (nf++ < 20) fprintf(stderr, "[FIOS-HLE] fallback to the SPU task (%s): src=%08X dst=%08X csz=%X n=%X w6=%08X first=%02X%02X\n",
                               why, src, dst, csz, n, w[6], vm_base[src - 2], vm_base[src - 1]);
    } else { InterlockedIncrement64(&s_fh_n); InterlockedAdd64(&s_fh_us, us); }
    const long long now = t1.QuadPart * 1000 / f.QuadPart;
    if (!s_fh_t0) s_fh_t0 = now;
    else if (now - s_fh_t0 >= 10000) {
        s_fh_t0 = now;
        fprintf(stderr, "[FIOS-HLE] 10s: %lld requests done natively (mean %.0f us), %lld sent to the SPU task\n",
                s_fh_n, s_fh_n ? (double)s_fh_us / s_fh_n : 0.0, s_fh_fb);
        s_fh_n = s_fh_us = s_fh_fb = 0;
    }
    return why ? 0 : 1;
}
s32 _cellSpursLFQueuePushBody(u64 queue_ea, u64 data_ea, u32 isBlocking)
{
    if (data_ea && fios_hle_try((uint32_t)queue_ea, (uint32_t)data_ea)) return CELL_OK;   /* v119 FIOS_HLE */
    ps3_fiosreq_push((uint32_t)queue_ea, (uint32_t)data_ea, vm_read32((uint32_t)queue_ea + 0x10));   /* v117 */
    if (ftl_line()) ps3_ftl(1, (uint32_t)queue_ea, 0);   /* v115 */
    if (!queue_ea || !data_ea) return CELL_SPURS_TASK_ERROR_NULL_POINTER;

    const uint32_t q = (uint32_t)queue_ea;
    const uint32_t data = (uint32_t)data_ea;
    const uint32_t size = vm_read32(q + 0x10);
    const uint32_t depth = vm_read32(q + 0x14);
    const uint32_t buffer = (uint32_t)vm_read64(q + 0x18) & ~1u;
    if (!size || !depth || !buffer) {
        static int bad_n = 0;
        if (bad_n++ < 16)
            fprintf(stderr,
                    "[cellSpurs] LFQueuePush INVALID q=0x%08X data=0x%08X "
                    "size=%u depth=%u buffer=0x%08X raw18=0x%016llX\n",
                    q, data, size, depth, buffer,
                    (unsigned long long)vm_read64(q + 0x18));
        return CELL_SPURS_TASK_ERROR_INVAL;
    }

    uint16_t position = vm_read16(q + 0x0E); /* push1.m_h8 */
    const uint16_t pop = vm_read16(q + 0x00); /* pop1.m_h1 */
    uint32_t used = position >= pop ? (uint32_t)(position - pop)
                                    : (uint32_t)(position + 2u * depth - pop);
    if (used >= depth) {
        static int full_n = 0;
        if (full_n++ < 16)
            fprintf(stderr,
                    "[cellSpurs] LFQueuePush FULL q=0x%08X position=%u pop=%u "
                    "used=%u depth=%u blocking=%u line=%08X/%08X/%08X/%08X\n",
                    q, position, pop, used, depth, isBlocking,
                    vm_read32(q + 0x00), vm_read32(q + 0x04),
                    vm_read32(q + 0x08), vm_read32(q + 0x0C));
        return CELL_SPURS_TASK_ERROR_AGAIN;
    }

    uint16_t next = (uint16_t)(position + 1u);
    if (next >= 2u * depth) next = 0;
    /* vm_write16 already takes the lock-line lock (see ef_line_end); the raw
     * store + ef_line_end is the same thing, spelled like the EF code. */
    spu_lockline_lock(); ef_w16(q + 0x0E, next); ef_line_end(q); /* reserve */
    atomic_thread_fence(memory_order_seq_cst);
    memcpy(vm_base + buffer + (size_t)(position % depth) * size,
           vm_base + data, size);
    atomic_thread_fence(memory_order_seq_cst);
    spu_lockline_lock(); ef_w16(q + 0x08, next); ef_line_end(q); /* push1.m_h5: publish completion */

    extern int spu_taskset_signal_parked_any(uint32_t wait_obj);
    /* The taskset channel normalizes its raw policy-module wait word to the
     * LFQueue base before publishing the parked object. */
    int woke = spu_taskset_signal_parked_any(q);
    static int n = 0;
    if (n++ < 32)
        fprintf(stderr,
                "[cellSpurs] LFQueuePush q=0x%08X slot=%u head=%u->%u pop=%u "
                "size=%u depth=%u woke=%d blocking=%u\n",
                q, (unsigned)(position % depth), position, next, pop,
                size, depth, woke, isBlocking);
    return CELL_OK;
}

s32 cellSpursLFQueueAttachLv2EventQueue(u64 queue_ea)
{
    static int _n = 0;
    if (_n++ < 8) printf("[cellSpurs] LFQueueAttachLv2EventQueue(q=0x%08X)\n", (u32)queue_ea);
    return CELL_OK;
}
