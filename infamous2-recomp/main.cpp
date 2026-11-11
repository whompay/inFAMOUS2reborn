/*
 * ps3recomp game project -- entry point template.
 *
 * This is the runner. It boots a lifted title through the toolkit's PPU
 * scaffold: allocate the flat guest VM, load the PPU ELF named on the command
 * line, register the lifted function table and the HLE NID handlers, start a
 * frame clock, and dispatch the entry OPD.
 *
 * The scaffold does the work; nothing below reimplements it. ppu_load_elf,
 * ppu_recomp_register, ppu_hle_init, ppu_sysprx_register, ppu_fs_register,
 * lv2_init_syscalls and ppu_run all come from the toolkit, compiled into this
 * project from PS3RECOMP_DIR (see CMakeLists.txt). What is left here is the
 * part a port owns: which backend to present through, how the frame clock is
 * paced, and whatever diagnostics the title turns out to need.
 *
 * runtime/ppu/tests/boot_main.cpp in the toolkit is the same boot in its
 * fully-instrumented form -- crash filter, hang watchdog, guest-PC sampling
 * profiler. Read it when a boot goes wrong; copy from it what the title
 * needs. It is deliberately not what a fresh project starts with.
 *
 * The one part that IS duplicated from it is the frame clock below, because a
 * port is expected to change its pacing and a shared one would be the wrong
 * shape for that. It was copied faithfully, comments and all. If a pacing bug
 * is fixed in one of the two, look at the other.
 *
 * This builds on Windows, macOS and Linux. CI builds and runs it against the
 * boot smoke title on the last two, and compile-checks it against clang-cl on
 * the first (tools/check_ppu_scaffold.py).
 */

/* The lifter's generated header, which declares ppu_context and the function
 * table. It comes first: everything below is written against that struct. */
#include "ppu_recomp.h"

/* The Win32 names -- Sleep, GetTickCount64, CreateThread, InterlockedIncrement,
 * VirtualAlloc, the scalar typedefs -- from one place. On Windows this is a
 * passthrough to <windows.h>; off Windows the toolkit's shim supplies the same
 * names over pthreads and mmap, so the call sites below are written once. */
#include "win32_compat.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
/* timeBeginPeriod: <windows.h> arrives with WIN32_LEAN_AND_MEAN set, which
 * excludes the multimedia timer API, so it is asked for by name -- and after
 * windows.h, since timeapi.h uses UINT. No shim exists off Windows because
 * there is nothing to shim: the 15.6 ms default timer granularity this works
 * around is a Windows problem. */
#include <timeapi.h>
#endif

/* ---------------------------------------------------------------------------
 * What the scaffold gives this file
 * -----------------------------------------------------------------------*/
extern "C" {
uint32_t ppu_load_elf(const char* path);      /* ELF -> guest RAM, returns entry OPD */
void     ps3_prof_main_start(void);           /* S180: v62 sampler (ppu_loader.cpp), PS3_PROF_MAIN */
void     ps3_ab_start(void);                  /* S180: v66 in-run A/B phases (ppu_loader.cpp), PS3_AB=1 */
void     ps3_qp_main_start(void);             /* QP: main-thread P-core pin / core reservation (ppu_loader.cpp) */
void     ppu_recomp_register(void);           /* generated: lifted table -> address map */
void     ppu_hle_init(void);                  /* firmware import NID -> HLE handler */
void     ppu_sysprx_register(void);           /* boot-critical CRT (sys_initialize_tls, ...) */
void     ppu_fs_register(void);               /* cellFs over the real game directory */
void     lv2_init_syscalls(void);             /* the lv2 syscall table */
int      ppu_run(uint32_t entry_opd, uint32_t stack_top);

/* This project's own: load whatever real system PRX modules the title needs
 * into guest RAM and register their exports. stubs.cpp has the empty version
 * a game with no lifted PRX wants. */
void     ps3_load_prx_modules(void);

extern const char* ppu_vfs_root;              /* host dir the PS3 mount points map into */

/* The status the guest handed to sys_process_exit. A title that exits that way
 * never returns through ppu_run, so this is only read on the path where the
 * entry function unwound instead. */
extern int     g_sys_process_exit_called;
extern int32_t g_sys_process_exit_code;

/* Host-provided symbols the runtime and the HLE libraries link against. */
uint8_t* vm_base = nullptr;
extern uint32_t ppu_vm_size;                  /* ppu_loader.cpp: the OOB guard */

/* Guest-callback dispatch. g_ps3_guest_caller is the hook the HLE runtime
 * calls back into recompiled code through -- cellSysutil events and the GCM
 * vblank/flip handlers. ppu_guest_call does the OPD -> dispatch. */
typedef void (*ps3_guest_caller_fn)(uint32_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t, uint64_t, uint64_t, uint64_t);
extern ps3_guest_caller_fn g_ps3_guest_caller;
uint64_t ppu_guest_call(uint32_t, uint64_t, uint64_t, uint64_t, uint64_t,
                        uint64_t, uint64_t, uint64_t, uint64_t);

/* The GCM half of the frame clock. */
void     cellGcmTickVBlank(void);
void     cellGcmTickFlip(void);
int      cellGcm_take_flip_pending(void);
void     cellGcm_rsx_process_fifo(void);      /* drain get -> put */
unsigned cellGcm_flip_request_count(void);
/* HP4A (2026-11-10): FIFO_KICK_LOOP + SPU_DMA_OPT switches (cellGcmSys.c / spu_channels.c). */
int      cellGcm_fifo_kick_loop_on(void);     /* FIFO_KICK_LOOP=0|1|ab (ABX 0x40000000) */
int      cellGcm_fifo_kick_wait2(unsigned ms);/* 1 = woken by a kick, 0 = timeout */
void     cellGcm_hp4a_kick_stats(unsigned long long* put_kicks, unsigned long long* dry_set,
                                 unsigned long long* dry_coalesced);
int      spu_dmaopt_now(void);                /* SPU_DMA_OPT effective mask now (ABX 0x20000000) */
extern volatile int g_abx;
}

/* ---------------------------------------------------------------------------
 * RSX present backend
 * -----------------------------------------------------------------------*/
/* D3D12 on Windows, Metal on Apple, and the null backend's headless software
 * path anywhere else -- the same selection runtime/host/host_posix.c and the
 * toolkit's boot harness make. All three expose the same three entry points,
 * so the frame clock below is backend-agnostic. */
#if defined(_WIN32)
extern "C" int  rsx_d3d12_backend_init(uint32_t w, uint32_t h, const char* title);
extern "C" void rsx_d3d12_backend_present(void);
extern "C" int  rsx_d3d12_backend_pump_messages(void);
#  define rsx_backend_init    rsx_d3d12_backend_init
#  define rsx_backend_present rsx_d3d12_backend_present
#  define rsx_backend_pump    rsx_d3d12_backend_pump_messages
#  define RSX_BACKEND_NAME    "D3D12"
#elif defined(__APPLE__)
extern "C" int  rsx_metal_backend_init(uint32_t w, uint32_t h, const char* title);
extern "C" void rsx_metal_backend_present(void);
extern "C" int  rsx_metal_backend_pump_messages(void);
#  define rsx_backend_init    rsx_metal_backend_init
#  define rsx_backend_present rsx_metal_backend_present
#  define rsx_backend_pump    rsx_metal_backend_pump_messages
#  define RSX_BACKEND_NAME    "Metal"
#else
extern "C" int  rsx_null_backend_init(uint32_t w, uint32_t h, const char* title);
extern "C" void rsx_null_backend_present(void);
extern "C" int  rsx_null_backend_pump_messages(void);
#  define rsx_backend_init    rsx_null_backend_init
#  define rsx_backend_present rsx_null_backend_present
#  define rsx_backend_pump    rsx_null_backend_pump_messages
#  define RSX_BACKEND_NAME    "null (headless software)"
#endif

/* ---------------------------------------------------------------------------
 * Configuration
 * -----------------------------------------------------------------------*/
/* The flat VM must span every region the PS3 memory map uses, not just the
 * game image: sys_ppu_thread_create puts thread stacks at 0xD0000000, so a
 * smaller arena makes every spawned thread's stack access out of bounds. */
#define VM_SIZE       0x100010000ull   /* the full 32-bit guest space + a 64K guard */
#define STACK_TOP     0x0FF00000u      /* main-thread stack, below the 0x10000000 segment */
#define WINDOW_WIDTH  1280
#define WINDOW_HEIGHT 720

/* ---------------------------------------------------------------------------
 * Frame clock
 * -----------------------------------------------------------------------*/
/* On real hardware the RSX raises a vblank interrupt about 60 times a second
 * and that is what drives the game's frame loop. With no RSX it is synthesized
 * here: a host thread calls cellGcmTickVBlank/TickFlip, which invoke the
 * handlers the guest registered, and drains the GCM FIFO. Without this the
 * title initialises, registers its handlers and then waits forever.
 *
 * The ticks are driven off real elapsed time rather than off how long present()
 * took. A hidden or occluded window makes present block hard, and pacing the
 * ticks behind it paces the whole game behind it. */
static volatile LONG g_frames_presented = 0;

#ifdef PORT_PGO_GEN
extern "C" int __llvm_profile_write_file(void);
static DWORD WINAPI pgo_profile_flush_thread(LPVOID)
{
    for (;;) {
        Sleep(30000);
        __llvm_profile_write_file();
    }
}
#endif

/* Frames handed to the backend at a guest FLIP boundary -- one per frame the
 * guest actually finished. The presents made before the guest's first flip, so
 * a fresh window is not left blank through a long boot, carry no guest frame
 * and are deliberately not counted. */
extern "C" unsigned ppu_boot_frames_presented(void)
{
    return (unsigned)g_frames_presented;
}

static void present_guest_frame(void)
{
    rsx_backend_present();
    /* This thread increments and guest threads read, so interlocked rather
     * than a volatile ++, which on arm64 is neither atomic nor a fence. */
    InterlockedIncrement(&g_frames_presented);
}

extern "C" void (*g_gcm_flip_sync_present)(void);
static void flip_sync_present(void) { present_guest_frame(); }

static DWORD WINAPI frame_clock(LPVOID)
{
    /* LX FLIP_SYNC_PRESENT (default 1): see cellGcmSys.c -- one present per guest flip even
     * when the drain is catching up. 0 = old (frames merged while backlogged). */
    { const char* e = getenv("FLIP_SYNC_PRESENT");
      if (!(e && *e == '0')) g_gcm_flip_sync_present = flip_sync_present;
      fprintf(stderr, "[FLIPSYNC] LX present at every FIFO flip %s (FLIP_SYNC_PRESENT)\n",
              g_gcm_flip_sync_present ? "ON" : "off"); }
    const char* title = getenv("PS3_TITLE");
    if (!title || !*title) title = "inFamous 2 Recomp";

    int rsx_ok = (rsx_backend_init(WINDOW_WIDTH, WINDOW_HEIGHT, title) == 0);
    fprintf(stderr, "[rsx] %s backend init %s\n", RSX_BACKEND_NAME,
            rsx_ok ? "OK -- window open" : "FAILED");

    unsigned  last_flip = 0;
    ULONGLONG next_tick = GetTickCount64();

    /* HP4A FIFO_KICK_LOOP=0 (default, old flat Sleep(4)) | 1 | ab (on while g_abx & 0x40000000):
     * the idle wait becomes a wait on the cellGcm kick event (put-register store / dry ref poll)
     * with the same 4 ms timeout, so it is never slower than the old sleep. The kick-pending flag is
     * cleared right after the wake, BEFORE this iteration's drain (cellGcm_fifo_kick_wait2), so a kick
     * made during the drain re-signals the event instead of being lost. [FIFOLOOP] every ~300 frames;
     * [HP4AB] phase markers whenever the FIFO / SPU_DMA_OPT switch state changes (same-process A/B). */
    LARGE_INTEGER fl_qf; QueryPerformanceFrequency(&fl_qf);
    LARGE_INTEGER fl_tp; QueryPerformanceCounter(&fl_tp);
    unsigned long long fl_kick = 0, fl_to = 0, fl_iter = 0, fl_bootskip = 0;
    LONGLONG fl_sleep = 0, fl_t_print = fl_tp.QuadPart, fl_boot_pres = 0;
    LONG fl_frames0 = g_frames_presented;
    int hp_key = -1; unsigned hp_phase = 0;

    for (;;) {
        const int kick_on = cellGcm_fifo_kick_loop_on();
        { const int dmo = spu_dmaopt_now();
          const int key = (kick_on << 16) | (dmo & 0xFFFF);
          if (key != hp_key) {
              fprintf(stderr, "[HP4AB] phase %u: fifo=%d dma=0x%X (abx 0x%X)\n", hp_phase++, kick_on, (unsigned)dmo, (unsigned)g_abx);
              hp_key = key;
          } }
        LARGE_INTEGER fl_t0; QueryPerformanceCounter(&fl_t0);
        if (kick_on) { if (cellGcm_fifo_kick_wait2(4)) fl_kick++; else fl_to++; }
        else         { Sleep(4); fl_to++; }
        LARGE_INTEGER fl_t1; QueryPerformanceCounter(&fl_t1);
        fl_sleep += fl_t1.QuadPart - fl_t0.QuadPart; fl_iter++;
        { const LONG fr = g_frames_presented;
          if (fr - fl_frames0 >= 300 || fl_t1.QuadPart - fl_t_print >= fl_qf.QuadPart * 30) {
              unsigned long long pk = 0, ds = 0, dc = 0;
              cellGcm_hp4a_kick_stats(&pk, &ds, &dc);
              const double wall = (double)(fl_t1.QuadPart - fl_t_print) / (double)fl_qf.QuadPart;
              fprintf(stderr, "[FIFOLOOP] HP4A kick=%d: %llu iters in %.1f s (%ld frames, %.1f iters/frame): woken by kick %llu, timeout %llu, "
                              "avg sleep %.0f us; cumulative put-kick SetEvent %llu, dry-poll SetEvent %llu issued / %llu coalesced; boot presents skipped %llu\n",
                      kick_on, fl_iter, wall, (long)(fr - fl_frames0), (fr - fl_frames0) > 0 ? (double)fl_iter / (double)(fr - fl_frames0) : 0.0,
                      fl_kick, fl_to, fl_iter ? (double)fl_sleep * 1e6 / (double)fl_qf.QuadPart / (double)fl_iter : 0.0,
                      pk, ds, dc, fl_bootskip);
              fl_kick = fl_to = fl_iter = 0; fl_sleep = 0; fl_t_print = fl_t1.QuadPart; fl_frames0 = fr;
          } }
        ULONGLONG now = GetTickCount64();

        int fired = 0;
        while ((long long)(now - next_tick) >= 0 && fired < 240) {
            cellGcmTickVBlank();
            cellGcmTickFlip();
            /* Present a pending flip BEFORE draining any further. The flip
             * fires at a get==put frame boundary on the guest thread, so the
             * batch held right now is exactly the completed frame; presenting
             * after the drain races the guest's next-frame writes and shows a
             * mixed one. */
            if (rsx_ok && cellGcm_take_flip_pending()) {
                present_guest_frame();
                last_flip = cellGcm_flip_request_count();
            }
            /* Drain the FIFO every tick. This is what writes the RSX sync-fence
             * labels the game's per-frame logic blocks on, so it has to keep
             * advancing at 60 Hz even while present() throttles. */
            if (rsx_ok) cellGcm_rsx_process_fifo();
            next_tick += 16;             /* ~60 Hz */
            fired++;
        }
        if (fired >= 240) next_tick = now;   /* fell too far behind -- resync */

        /* Drain at the outer cadence too, not only on the 16 ms tick: titles
         * fence every render pass on a label the drain writes, and one wave of
         * those at 16 ms apiece paces the guest into single-figure frame rates.
         * The real RSX writes them in microseconds. */
        if (rsx_ok) {
            if (cellGcm_take_flip_pending()) {
                present_guest_frame();
                last_flip = cellGcm_flip_request_count();
            }
            cellGcm_rsx_process_fifo();

            if (rsx_backend_pump() != 0) {
                rsx_ok = 0;              /* window closed */
                continue;
            }
            /* Present on a guest flip. A present on a fixed clock can catch the
             * drain mid-frame and flash a partial one. Before the first flip
             * present freely, so the window is not blank during boot. */
            unsigned fc = cellGcm_flip_request_count();
            if (fc != last_flip) {
                present_guest_frame();
                last_flip = fc;
            } else if (fc == 0) {
                /* HP4A: with kick wakes this loop can spin far faster than the old 4 ms; keep the
                 * pre-first-flip boot presents at the old cadence. */
                LARGE_INTEGER bq; QueryPerformanceCounter(&bq);
                if (!kick_on || bq.QuadPart - fl_boot_pres >= fl_qf.QuadPart / 250) {
                    fl_boot_pres = bq.QuadPart;
                    rsx_backend_present();
                } else fl_bootskip++;
            }
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * VFS root
 * -----------------------------------------------------------------------*/
/* Derive the directory holding PS3_GAME from the EBOOT path
 * <root>/PS3_GAME/USRDIR/EBOOT.elf -> <root>. $PS3_VFS_ROOT overrides. */
static char s_vfs_root[1024];

static void derive_vfs_root(const char* eboot)
{
    const char* env = getenv("PS3_VFS_ROOT");
    if (env && *env) { ppu_vfs_root = env; return; }

    strncpy(s_vfs_root, eboot, sizeof s_vfs_root - 1);
    for (char* p = s_vfs_root; *p; p++) if (*p == '\\') *p = '/';
    /* strip EBOOT.elf, USRDIR and PS3_GAME */
    for (int i = 0; i < 3; i++) { char* s = strrchr(s_vfs_root, '/'); if (s) *s = 0; }
    if (!s_vfs_root[0]) strcpy(s_vfs_root, ".");
    ppu_vfs_root = s_vfs_root;
}

/* ---------------------------------------------------------------------------
 * Guest VM
 * -----------------------------------------------------------------------*/
#ifdef _WIN32
/* Demand-paging for the flat VM: reserve the whole 4 GB guest space, which
 * costs no commit, and commit each 64 KB page on first access. Every 32-bit
 * guest offset is then valid, so a garbage guest pointer reads zero instead of
 * killing the process. Faults outside the arena fall through untouched. */
static LONG WINAPI vm_commit_veh(EXCEPTION_POINTERS* ep)
{
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        ULONG_PTR fault = ep->ExceptionRecord->ExceptionInformation[1];
        uintptr_t base  = (uintptr_t)vm_base;
        if (vm_base && fault >= base && fault < base + VM_SIZE) {
            void* page = (void*)(fault & ~(uintptr_t)0xFFFF);
            if (VirtualAlloc(page, 0x10000, MEM_COMMIT, PAGE_READWRITE))
                return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

/* S180: crash reporter. This runner had no exception filter, so a host crash ended the
 * process with no log line (P-drive2: silent exit at 265 s). Unhandled exceptions now log
 * code / RVA / fault address / host backtrace RVAs (resolve with build-fast\inFamous2Recomp.map)
 * and write a small minidump next to the run's frame dumps (CELLMARK_DUMP_DIR, else cwd);
 * exit() and abort() log a line too. S180_CRASH=0 disables it. */
#ifdef _WIN32
#include <dbghelp.h>
#include <signal.h>
#pragma comment(lib, "dbghelp.lib")
static void s180_backtrace(const char* tag, void* self_addr)
{
    void* frames[48];
    USHORT n = RtlCaptureStackBackTrace(0, 48, frames, NULL);
    HMODULE self = NULL;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)self_addr, &self);
    for (USHORT i = 0; i < n; i++) {
        HMODULE m = NULL;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)frames[i], &m);
        if (m == self)
            fprintf(stderr, "%s   #%-2u rva=0x%llX\n", tag, i, (unsigned long long)((char*)frames[i] - (char*)m));
        else
            fprintf(stderr, "%s   #%-2u %p (module %p)\n", tag, i, frames[i], (void*)m);
    }
}
static void s180_minidump(EXCEPTION_POINTERS* ep)
{
    char path[1100];
    const char* dir = getenv("CELLMARK_DUMP_DIR");
    snprintf(path, sizeof path, "%s%scrash-%lu.dmp", dir && *dir ? dir : ".", "\\", (unsigned long)GetCurrentProcessId());
    HANDLE f = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) { fprintf(stderr, "[S180-CRASH] minidump: cannot create %s\n", path); return; }
    MINIDUMP_EXCEPTION_INFORMATION mei;
    mei.ThreadId = GetCurrentThreadId(); mei.ExceptionPointers = ep; mei.ClientPointers = FALSE;
    BOOL ok = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), f,
                                (MINIDUMP_TYPE)(MiniDumpWithThreadInfo | MiniDumpWithIndirectlyReferencedMemory),
                                ep ? &mei : NULL, NULL, NULL);
    CloseHandle(f);
    fprintf(stderr, "[S180-CRASH] minidump %s: %s\n", ok ? "written" : "FAILED", path);
}
static volatile LONG s180_in_crash = 0;
static LONG WINAPI s180_crash_filter(EXCEPTION_POINTERS* ep)
{
    if (InterlockedExchange(&s180_in_crash, 1)) return EXCEPTION_EXECUTE_HANDLER;
    EXCEPTION_RECORD* er = ep->ExceptionRecord;
    HMODULE mod = NULL;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)er->ExceptionAddress, &mod);
    fprintf(stderr, "\n[S180-CRASH] unhandled exception code=0x%08lX at %p (module %p rva 0x%llX) host thread %lu\n",
            (unsigned long)er->ExceptionCode, er->ExceptionAddress, (void*)mod,
            (unsigned long long)((char*)er->ExceptionAddress - (char*)mod), (unsigned long)GetCurrentThreadId());
    if (er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && er->NumberParameters >= 2)
        fprintf(stderr, "[S180-CRASH] %s fault address 0x%llX (vm_base %p)\n",
                er->ExceptionInformation[0] == 1 ? "write" : er->ExceptionInformation[0] == 8 ? "execute" : "read",
                (unsigned long long)er->ExceptionInformation[1], (void*)vm_base);
    s180_backtrace("[S180-CRASH]", (void*)&s180_crash_filter);
    fflush(stderr);
    s180_minidump(ep);
    fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}
/* Stack overflow cannot reach the unhandled filter reliably: log it first-chance (cheap test). */
static LONG WINAPI s180_fatal_veh(EXCEPTION_POINTERS* ep)
{
    DWORD c = ep->ExceptionRecord->ExceptionCode;
    if (c == EXCEPTION_STACK_OVERFLOW) {
        static volatile LONG once = 0;
        if (!InterlockedExchange(&once, 1)) {
            HMODULE mod = NULL;
            GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)ep->ExceptionRecord->ExceptionAddress, &mod);
            fprintf(stderr, "\n[S180-CRASH] STACK OVERFLOW at %p (rva 0x%llX) host thread %lu\n",
                    ep->ExceptionRecord->ExceptionAddress,
                    (unsigned long long)((char*)ep->ExceptionRecord->ExceptionAddress - (char*)mod),
                    (unsigned long)GetCurrentThreadId());
            fflush(stderr);
        }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
static void s180_abort_handler(int)
{
    fprintf(stderr, "\n[S180-CRASH] abort() / SIGABRT on host thread %lu\n", (unsigned long)GetCurrentThreadId());
    s180_backtrace("[S180-CRASH]", (void*)&s180_abort_handler);
    fflush(stderr);
    _exit(3);
}
static void s180_atexit(void)
{
    fprintf(stderr, "[S180] process exit() called (host thread %lu)\n", (unsigned long)GetCurrentThreadId());
    s180_backtrace("[S180]", (void*)&s180_atexit);
    fflush(stderr);
}
static void s180_install_crash_report(void)
{
    const char* e = getenv("S180_CRASH");
    if (e && *e == '0') return;
    ULONG guarantee = 64 * 1024;           /* room for the filter on the main thread */
    SetThreadStackGuarantee(&guarantee);
    SetUnhandledExceptionFilter(s180_crash_filter);
    AddVectoredExceptionHandler(0, s180_fatal_veh);
    signal(SIGABRT, s180_abort_handler);
    atexit(s180_atexit);
    fprintf(stderr, "[S180] crash reporter on (unhandled filter + minidump, SIGABRT, exit; S180_CRASH=0 off)\n");
}
#else
static void s180_install_crash_report(void) {}
#endif

static bool alloc_guest_vm(void)
{
#ifdef _WIN32
    AddVectoredExceptionHandler(1, vm_commit_veh);
    vm_base = (uint8_t*)VirtualAlloc(NULL, VM_SIZE, MEM_RESERVE, PAGE_READWRITE);
    ppu_vm_size = 0;              /* the whole space is backed; no OOB guard needed */
#else
    /* No vectored exception handlers off Windows, so the arena is committed up
     * front and lazily backed by the OS: only pages the title touches cost
     * anything. It stops below the 0xE0000000 mark, so ppu_vm_size arms the
     * loader's bounds check for what is left. */
    vm_base = (uint8_t*)calloc(1, 0xE0000000u);
    ppu_vm_size = 0xE0000000u;
#endif
    return vm_base != nullptr;
}

/* ---------------------------------------------------------------------------
 * Main
 * -----------------------------------------------------------------------*/
static void harness_guest_caller(uint32_t opd, uint64_t a0, uint64_t a1,
                                 uint64_t a2, uint64_t a3, uint64_t a4,
                                 uint64_t a5, uint64_t a6, uint64_t a7)
{
    ppu_guest_call(opd, a0, a1, a2, a3, a4, a5, a6, a7);
}

/* Title defaults remain externally overridable for diagnostics and A/B runs. */
static void set_default_env(const char* key, const char* value)
{
    if (getenv(key)) return;
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 0);
#endif
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("usage: %s <PPU ELF>\n", argv[0]);
        return 2;
    }

    s180_install_crash_report();

    /* Image 2 is the RAD/Bink raw-SPU worker. It binds its request queue and
     * waits at sys_spu_thread_receive_event (stop 0x110); serving that wait on
     * the lifted SPU host thread keeps decode native and lets it publish its
     * own completion events. SPU_LIFTED_RECV=0 restores the legacy park. */
    set_default_env("SPU_LIFTED_RECV", "1");
    /* Q (2026-10-16): w2 BatchJobList is the tutorial frame's latency-critical
     * five-lane SPU workload. On this hybrid host, a same-process stationary
     * tutorial A/B of all five lanes on P-cores improved 16.96 -> 18.92 FPS
     * across seven matched 25-second phases (+11.5%). Keep other workloads free to
     * use E-cores; SPURS_PCORE_WIDS=- restores unrestricted scheduling. */
    set_default_env("SPURS_PCORE_WIDS", "2");
    /* S180: native RAD/Bink decode is the default again. The "hang" was never a
     * mis-lift: the SP Movie Player's buffer handshake (lwcond 0xA08828, wait at
     * lr 0x1D2754, see PS3_LWCOND_STRICT_LR below) woke on the 1 ms poll before
     * the main thread published the frame-pool EA, so the YUV planes were built
     * from offset 0, which GCM_LOW_IDENTITY turned into EA 0x40400000 = the GCM
     * command buffer. The worker then overwrote the FIFO with luma, the walker
     * resynced past the frame's reference and the main thread waited forever.
     * INFAMOUS_BINK_BYPASS=1 restores the completion-only bypass. */
    set_default_env("INFAMOUS_BINK_BYPASS", "0");
    /* EDGE zlib can report one transient queue-completion error during the
     * load-menu handoff. Keep its persistent worker alive at that exact lifted
     * assertion site; 0 restores the retail stop-0x3FFF path for A/B tests. */
    set_default_env("INFAMOUS_EDGEZLIB_CONTINUE", "1");
    /* The current world path completes its 1280x720 deferred frame in an
     * offscreen colour target but never emits the expected display composite.
     * Present that final target only when the generic stall detector sees a
     * substantial offscreen frame; OFFSCREEN_STALL_PRESENT=0 is the A/B escape. */
    set_default_env("OFFSCREEN_STALL_PRESENT", "1");
    /* R170: the real display post (fp 6133E4AD) was correct; the black came from the
     * fullscreen fade overlay drawn after it (alpha ~0.98, clamped by the backend's
     * FADE_FIX when held). Present the game's own post + UI; POST_FALLBACK=1 brings
     * back the raw-scene RT 0x011C0000 override (it also needs POST_REAL=0). */
    set_default_env("POST_FALLBACK", "0");
    /* R171: the deferred lights carry depth-bounds windows ~3e-5 wide near z 0.9994
     * ("[RSX] depth bounds ON [0.999397, 0.999429]"); our stored depth does not match the
     * PS3's closely enough, so the test rejected nearly every light pixel and the lit scene
     * held only emissive windows. The light shaders attenuate by distance themselves, so the
     * test is an optimisation here. DEPTH_BOUNDS=1 restores it for A/B work. */
    set_default_env("DEPTH_BOUNDS", "0");
    /* LX (2026-10-16): the sun/ambient pass (fph 9721DEC8) and the shadow resolve read the guest
     * STENCIL from B of the depth-as-A8R8G8B8 copy (D24S8 word = depth<<8 | stencil). Bit 7
     * (characters, sref 128) selects full vs 0.35 ambient, bit 6 overrides the SPU light mask.
     * Without it Cole/NPCs were black silhouettes. Verified same-process A/B test-runs/LX-zst-*.
     * ZCONV_STENCIL=0 = old. */
    set_default_env("ZCONV_STENCIL", "1");
    /* LX (2026-10-16): frames end ONLY at the guest's FIFO flip. The display-clear boundary
     * heuristic presented mid-frame here (the display buffer doubles as G-buffer MRT0 and is
     * cleared twice per frame) -> yellow G-buffer frames, ghosting and dark frames at camera
     * cuts. Pairs with FLIP_SYNC_PRESENT (frame clock). RSX_ACCUM_FRAME=0 = old. */
    set_default_env("RSX_ACCUM_FRAME", "1");
    /* LX: the real post/UI renders correctly now; the R170 black guard swapped in the raw scene RT
     * (no bloom, no UI) after any 12 black probes (fades, cuts) and back 64 probes later. */
    set_default_env("POST_AUTO_BLACK", "0");
    /* F182 (perf): bring-up diagnostics used to remain active in normal play.
     * BLACK_PROBE copied the entire swapchain to a readback buffer every eight
     * presents; R170_EARLYCAP automatically performed four full rsxcap dumps;
     * R170_DISPLOG and MAINTEX_LOG repeatedly scanned world draw/texture data.
     * They are valuable on demand but should not consume bandwidth, CPU time,
     * or cause multi-second capture hitches in a performance build. */
    set_default_env("BLACK_PROBE", "0");
    set_default_env("R170_EARLYCAP", "0");
    set_default_env("R170_DISPLOG", "0");
    set_default_env("MAINTEX_LOG", "0");
    /* M180 (2026-10-15): indexed TRIANGLE_STRIP expansion alternates the winding of odd triangles
     * like the non-indexed path (backend M180_STRIPWIND); before, half of every indexed strip had
     * reversed facing under guest culling. Correctness fix; NOT what blacked out the start screen
     * (see M180_SKYZ). M180_STRIPWIND=0 = old expansion. */
    set_default_env("M180_STRIPWIND", "1");
    /* M180 start-screen fix: the G-buffer backdrop strip bound to the 1x1 texture at RSX offset
     * 0x01F59980 is front-facing in our replay but back-facing (culled) on the PS3; drawn, it covers
     * the frame at depth ~0.9985 and the z = 1.0 cloud pass fails -> black start screen. The backend
     * reverses that draw's facing. M180_SKYZ=0 disables (A/B). */
    set_default_env("M180_SKYZ", "01F59980");
    /* R171: Wwise job-chain completion events from the jobs' own mailbox writes (one event per
     * "notify" job to the queue bound to its port) instead of a broadcast after every job; the
     * broadcast desynchronised the audio thread and caused the title/intro command-ring stalls.
     * SPURS_JC_EVENTS=broadcast restores the old behaviour. */
    set_default_env("SPURS_JC_EVENTS", "mbox");
    /* The world renderer terminates SPU-built FIFO segments with a self-JUMP
     * and patches it once the continuation is ready.  PUT still points into
     * the primary ring during that hardware wait, so following PUT immediately
     * drops the continuation's label-0x85 completion fence. */
    set_default_env("GCM_WAIT_SELF_JUMP", "1");
    /* The two SP Movie worker loops wait once for their main-thread handoff
     * and then consume the pool/status fields.  A polling lwcond can wake
     * spuriously after 1 ms and lets them run with the sentinel texture offset
     * 0xFFFFFFFF.  Keep FIOS on the compatibility polling model, but make the
     * two title-specific movie waits real signal-driven waits. */
    /* S180: + 0x1D2754, the SP Movie Player's frame-pool request (player+0x48 = 1, wait on
     * lwcond 0xA08828 until the main thread stores the pool EA at player+0x4C and sets +0x48
     * = 2). The guest re-checks only "+0x48 != 0" after the wait, so a poll wake builds the
     * movie texture/DMA addresses from a zero base. */
    set_default_env("PS3_LWCOND_STRICT_LR", "1D09B8,1D2924,1D2754");
    /* These are the loader/player handoff objects used by the strict sites
     * above. Preserve one early signal if host scheduling lets the producer
     * reach either object just before its movie worker begins waiting. */
    set_default_env("PS3_LWCOND_STICKY", "A08778,A08780");
    set_default_env("GCM_AUTOMAP_SYSMEM", "1");
    /* This title's statically-linked movie helper indexes the first published
     * GCM offset table with zero-based offsets into its two-frame YUV pool.
     * Real firmware leaves usable identity translations in that low IO range;
     * without them the helper stores 0xFFFFFFFF texture offsets and the intro
     * collapses to a flat green draw. Keep this title quirk local and allow an
     * explicit GCM_LOW_IDENTITY=0 override for A/B testing. */
    /* S180: the zero-based offsets were the lwcond race above, not firmware behaviour; with
     * the strict wait the helper sees the real pool EA (0x406xxxxx) and indexes the normal
     * EA->IO entries. The low identity entries made a zero base alias the command buffer.
     * GCM_LOW_IDENTITY=1 restores them for A/B work. cellGcmSys tests only presence, so
     * this title no longer sets it and an explicit "0" is removed here. */
    if (const char* li = getenv("GCM_LOW_IDENTITY")) {
        if (li[0] == '0' && li[1] == 0) {
#ifdef _WIN32
            _putenv_s("GCM_LOW_IDENTITY", "");
#else
            unsetenv("GCM_LOW_IDENTITY");
#endif
        }
    }
    /* P172: the title/intro freeze was an unlifted Wwise SPU job (EBOOT VA 0x88CB80, 6032 bytes,
     * fp E557DC3BB8031540): every dispatch MISSed, the chain skipped it, the voice never got its
     * output and the AK thread re-submitted it ~5k times/s while the main thread blocked on the
     * full Wwise command ring.  spu/raw_jobs_register.c now registers the 20 remaining EBOOT job
     * blobs (images 310..329; P172_RAWJOBS=0 = old set) under their exact fingerprint and the
     * trailing-zero-trimmed one; SPU_JOB_TRIM_MATCH=1 lets a dispatch with a different blob size
     * find the trimmed alias (0 = exact fingerprints only). */
    set_default_env("SPU_JOB_TRIM_MATCH", "1");
    /* C173: SpuLight (w2) was the frame limiter: one host lane ran ~9.4 s of every 10 s while
     * the main thread waited at lr 001DB7E8 (~5 fps in the tutorial). Running its 5 virtual
     * lanes concurrently (playbook 7.7) measured 5 -> ~14 fps; adding w0,1,3,4 gave nothing
     * more. SPURS_PAR_LANES=99 restores serial lanes. */
    set_default_env("SPURS_PAR_LANES", "2");
    /* S180: the toolkit's v141 audio-core rule (an inFamous-1 setting) reserves a whole P-core
     * (2 logical CPUs, every other thread kept off it) for the SPURS instance-1 kernel thread. In
     * this title that thread stays under 2% CPU (S-ab0: never in [TELEM-CPU] >= 2%), so the core
     * idled while SpuLight lanes and the main thread shared the rest. S-ab2pcaud vs S-ab0, matched
     * tutorial windows: ~+9% fps. PS3_AUDIO_CORE=1 restores the reservation. */
    set_default_env("PS3_AUDIO_CORE", "0");
    /* F181 (perf): quiet per-event logging unless asked for. ps3_log.h makes a REDIRECTED stderr
     * verbose by default, and the per-wait lines from every hot guest thread then serialise on one
     * FILE lock (run.cmd notes ~1-2 fps). run.cmd/session.ps1 already pass 0; this covers any other
     * launcher that captures stderr. PS3_VERBOSE=1 still turns the full log on. */
    set_default_env("PS3_VERBOSE", "0");
    /* SPURS_PMTIME takes clocks and interlocked counters in the lane hot paths.
     * Keep it diagnostic-only even when the executable is launched directly
     * instead of through run.cmd; SPURS_PMTIME=1 remains an explicit opt-in. */
    set_default_env("SPURS_PMTIME", "0");

#ifdef _WIN32
#pragma comment(lib, "winmm.lib")
    /* 1 ms timer resolution. The default granularity is about 15.6 ms, which
     * inflates every shorter wait the title makes and throttles the whole
     * thing. POSIX timers are already fine-grained. */
    timeBeginPeriod(1);
    setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered: do not lose prints on a kill */
#endif

    printf("=== inFamous 2 recomp (BCES01143) ===\n");
    /* v170P: say which code-generation flags this build really used (the cached
     * CMAKE_*_FLAGS_RELEASE were empty, so "Release" used to mean /Od here). */
#if defined(__OPTIMIZE__)
    const int p170_opt = 1;
#else
    const int p170_opt = 0;
#endif
#if defined(__AVX2__) && defined(__FMA__)
    const int p170_avx2 = 1;
#else
    const int p170_avx2 = 0;
#endif
    fprintf(stderr, "[S180] movie defaults: INFAMOUS_BINK_BYPASS=%s PS3_LWCOND_STRICT_LR=%s GCM_LOW_IDENTITY=%s\n",
            getenv("INFAMOUS_BINK_BYPASS"), getenv("PS3_LWCOND_STRICT_LR"), getenv("GCM_LOW_IDENTITY") ? getenv("GCM_LOW_IDENTITY") : "(unset)");
    fprintf(stderr, "[P170] main/scaffold build: optimised=%d avx2+fma=%d (CMake PORT_OPT / PORT_HOST_ISA)\n",
            p170_opt, p170_avx2);

#ifdef PORT_PGO_GEN
    CreateThread(NULL, 0, pgo_profile_flush_thread, NULL, 0, NULL);
    fprintf(stderr, "[PGO] instrumented build: flushing LLVM_PROFILE_FILE every 30 seconds\n");
#endif

    if (!alloc_guest_vm()) {
        fprintf(stderr, "ERROR: could not allocate the guest address space\n");
        return 1;
    }

    uint32_t entry = ppu_load_elf(argv[1]);
    if (!entry) {
        fprintf(stderr, "ERROR: could not load %s\n", argv[1]);
        return 1;
    }

    derive_vfs_root(argv[1]);
    printf("[boot] VFS root: %s\n", ppu_vfs_root);

    ppu_recomp_register();   /* lifted function table -> address map */
    ps3_load_prx_modules();  /* this game's lifted system PRX, if it has any */
    ppu_hle_init();          /* firmware import NID -> HLE handlers */
    ppu_sysprx_register();   /* boot-critical CRT */
    ppu_fs_register();       /* cellFs over the game directory */
    lv2_init_syscalls();     /* the lv2 syscall table */

    /* Install the guest-callback hook, then start the frame clock. It no-ops
     * until the title registers its vblank and flip handlers during init. */
    g_ps3_guest_caller = harness_guest_caller;
    CreateThread(NULL, 4u * 1024 * 1024, frame_clock, NULL, 0, NULL);

    printf("\n[boot] dispatching entry OPD 0x%08X (stack top 0x%08X)\n\n",
           entry, STACK_TOP);

    /* S180: the toolkit's v62 sampling profiler ([PROF]/[PROF-FN] main|spu|fifo every 10 s,
     * resolve RVAs with build-fast\inFamous2Recomp.map) was never started by this runner, so
     * PS3_PROF_MAIN=<ms> did nothing. Start it on the main guest thread (it samples this thread). */
    {
        if (getenv("PS3_PROF_MAIN")) ps3_prof_main_start();
        /* ...and the v66 in-run A/B phase thread (PS3_AB=1, PS3_AB_SEQ / PS3_ABX_SEQ / PS3_AB_SEC;
         * [AB]/[ABX] phase markers in stderr), also never started by this runner. */
        if (getenv("PS3_AB")) ps3_ab_start();
    }
    /* ---- QP block (perf push 2026-10-16) ----
     * Main guest thread scheduling: QP_MAIN_PCORE / QP_MAIN_CORE (=1 on, =ab follow PS3_ABX_SEQ bits
     * 0x10000 / 0x20000). Unset = old behaviour (NORMAL priority, any CPU). Defaults go here after the A/B. */
    ps3_qp_main_start();
    /* ---- end QP block ---- */
    int rc = ppu_run(entry, STACK_TOP);
#ifdef PORT_PGO_GEN
    __llvm_profile_write_file();
#endif
    printf("\n[boot] ppu_run returned %d (entry function unwound)\n", rc);

    /* A title that called sys_process_exit never reaches this line: that path
     * ends in the host exit(). Getting here means the entry function returned
     * instead, so hand back the status the guest published if it published
     * one. Returning a hardcoded 0 would report success for a run that never
     * got anywhere. */
    return g_sys_process_exit_called ? (int)g_sys_process_exit_code : 0;
}
