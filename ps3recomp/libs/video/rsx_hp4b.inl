/* ===== HP4B (2026-11-10): render-side CPU offload / wait hygiene. #included by rsx_d3d12_backend.c right after
 * nl_ab_flip().  EVERYTHING here is env-gated and default OFF; with no HP4B env var set the backend behaves exactly
 * as HP3RSX v2 (only a few predictable branches are added).
 *
 *  PERF=1                 [HP4BPERF] per-step timers per NL_AB phase, ms per PRESENTED frame:
 *                         rot = BLIT_DEFER G-buffer RGBA->ARGB rotate (SPU lane thread), snap = TEX_SNAP memcpy (FIFO thread),
 *                         dec = rsx_texture_decode into the WC upload ring (render thread), pres = Present+move_to_next_frame,
 *                         gpuw = wait_for_gpu at the top of every replay, plus WAIT_SPIN hit/miss counters and SNAPUP direct copies.
 *  BLIT_CONV_MT=<n>       (1..7, default 0) split the deferred G-buffer rotate across n helper threads + the calling thread.
 *                         NL_AB=BLITCONV inverts it (phase B: n=0 <-> n, n defaults to 3 when unset). BLIT_CONV_MT_VERIFY=1
 *                         re-runs the single-thread rotate into a scratch buffer and memcmp()s ([HP4B] conv verify ... mismatches).
 *  WAIT_SPIN_US=<us>      (default 0) poll fences/events for up to <us> before blocking: wait_for_gpu, move_to_next_frame,
 *                         BLIT_DEFER fence, rf_wait_pumping. NL_AB=WAITSPIN inverts (150 us when unset).
 *  TEX_SNAP_UPLOAD=1      the TEX_SNAP arena of each parity is a persistently mapped CPU-cached (write-back, L0) D3D12 buffer;
 *                         linear A8R8G8B8 (0x85) snapshot textures whose rows are 256-byte multiples are copied by the GPU straight
 *                         from it (no rsx_texture_decode memcpy into the upload ring). Bytes are identical (that decode path IS a
 *                         row memcpy). TEX_SNAP_UPLOAD_MB=<n> (default 32) per parity; overflow falls back to the malloc arena.
 *                         NL_AB=SNAPUP inverts (decided per hand-off, so a phase flip never mixes arenas within a frame).
 *  HP4B_HEAP_PROBE=1      one-shot: does ID3D12Device3::OpenExistingHeapFromAddress accept vm_base / a private VirtualAlloc?
 *                         ([HP4B] heap probe ...). Wrapped heaps are released immediately; nothing uses them.
 * A marker "[HP4B] phase <k>: A|B" is printed at every NL_AB phase change (python3 _xfer/HP4A_phase.py HP4B stderr.log). */
extern volatile int g_hp4b_ab_phase;   /* cellSpurs.c: NL_AB phase mirror for the SPURS/audio switches */

static LONGLONG h4_qpc(void) { LARGE_INTEGER c; QueryPerformanceCounter(&c); return c.QuadPart; }
static LONGLONG h4_qpf(void) { static LONGLONG f = 0; if (!f) { LARGE_INTEGER q; QueryPerformanceFrequency(&q); f = q.QuadPart; } return f; }
enum { H4_ROT = 0, H4_SNAP, H4_DEC, H4_PRES, H4_GPUW, H4_SPINHIT, H4_SPINMISS, H4_DIRECT, H4_NSTEP };
static volatile LONG64 s_h4_t[2][H4_NSTEP], s_h4_n[2][H4_NSTEP];
static int h4_perf(void) { return perf_on(); }
static void h4_add(int k, LONGLONG dt)
{
    const int ph = nl_ab_phase() & 1;
    InterlockedExchangeAdd64(&s_h4_t[ph][k], dt);
    InterlockedIncrement64(&s_h4_n[ph][k]);
}
/* Render thread, once per replay at the Present site. */
static void h4_frame_end(int presented)
{
    static int last_ph = -1; static u32 k = 0;
    const int ph = nl_ab_phase() & 1;
    if (nl_ab_period()) {
        g_hp4b_ab_phase = ph;
        if (ph != last_ph) { fprintf(stderr, "[HP4B] phase %u: %s%c", k++, ph ? "B (NL_AB features inverted)" : "A", 10); last_ph = ph; }
    }
    if (!h4_perf() || !presented) return;
    static u64 pf[2]; static u32 n = 0;
    pf[ph]++;
    if (++n % 600u) return;
    const double ms = 1000.0 / (double)h4_qpf();
    char b[1024]; int l = 0;
    for (int p = 0; p < 2; p++) {
        if (!pf[p]) continue;
        LONG64 t[H4_NSTEP], c[H4_NSTEP];
        for (int i = 0; i < H4_NSTEP; i++) { t[i] = InterlockedExchange64(&s_h4_t[p][i], 0); c[i] = InterlockedExchange64(&s_h4_n[p][i], 0); }
        const double f = (double)pf[p];
        l += snprintf(b + l, sizeof b - (size_t)l,
                      " | %c %llu fr: rot %.3f (%.1f/fr) snap %.3f dec %.3f (%.1f/fr) pres %.3f gpuw %.3f ms/fr; spin hit %lld miss %lld; snapup direct %lld",
                      p ? 'B' : 'A', (unsigned long long)pf[p],
                      t[H4_ROT] * ms / f, c[H4_ROT] / f, t[H4_SNAP] * ms / f, t[H4_DEC] * ms / f, c[H4_DEC] / f,
                      t[H4_PRES] * ms / f, t[H4_GPUW] * ms / f,
                      (long long)c[H4_SPINHIT], (long long)c[H4_SPINMISS], (long long)c[H4_DIRECT]);
        if (l >= (int)sizeof b) l = (int)sizeof b - 1;
        pf[p] = 0;
    }
    fprintf(stderr, "[HP4BPERF] last 600 presented frames%s%c", b, 10);
}

/* ---- WAIT_SPIN_US ---- */
static int h4_wait_spin_us(void)
{
    static int v = -1, ab = -1;
    if (v < 0) { const char* e = getenv("WAIT_SPIN_US"); v = e ? atoi(e) : 0; if (v < 0) v = 0; if (v > 5000) v = 5000;
                 fprintf(stderr, "[HP4B] WAIT_SPIN_US=%d (spin-then-block fence/event waits)%c", v, 10); }
    const int on = nl_ab_flip(v > 0, &ab, "WAITSPIN");
    return on ? (v > 0 ? v : 150) : 0;
}
static int h4_spin_fence(ID3D12Fence* f, UINT64 val)
{
    const int us = h4_wait_spin_us();
    if (!us || !f) return 0;
    const LONGLONG dl = h4_qpc() + (LONGLONG)us * h4_qpf() / 1000000;
    for (;;) {
        if (f->lpVtbl->GetCompletedValue(f) >= val) { if (h4_perf()) h4_add(H4_SPINHIT, 0); return 1; }
        for (int z = 0; z < 16; z++) _mm_pause();
        if (h4_qpc() >= dl) break;
    }
    if (h4_perf()) h4_add(H4_SPINMISS, 0);
    return 0;
}
/* Returns 1 if ev became signalled during the spin (an auto-reset event is consumed, exactly as the blocking wait would). */
static int h4_spin_event(HANDLE ev)
{
    const int us = h4_wait_spin_us();
    if (!us || !ev) return 0;
    const LONGLONG dl = h4_qpc() + (LONGLONG)us * h4_qpf() / 1000000;
    for (;;) {
        if (WaitForSingleObject(ev, 0) == WAIT_OBJECT_0) { if (h4_perf()) h4_add(H4_SPINHIT, 0); return 1; }
        for (int z = 0; z < 64; z++) _mm_pause();
        if (h4_qpc() >= dl) break;
    }
    if (h4_perf()) h4_add(H4_SPINMISS, 0);
    return 0;
}

/* ---- BLIT_CONV_MT: row-parallel rsx_blit_rgba_to_argb_rows ---- */
typedef struct { const u8* s; size_t sp; u8* d; size_t dp; u32 w, h; } H4ConvJob;
#define H4C_MAXJOBS 512
static H4ConvJob s_h4c_job[H4C_MAXJOBS];
static volatile LONG s_h4c_njob = 0, s_h4c_next = 0, s_h4c_done = 0;
static HANDLE s_h4c_sem = NULL;
static int s_h4c_nthr = 0;
static void h4c_drain(void)
{
    for (;;) {
        const LONG i = InterlockedIncrement(&s_h4c_next) - 1;
        if (i >= s_h4c_njob) return;
        const H4ConvJob* j = &s_h4c_job[i];
        rsx_blit_rgba_to_argb_rows(j->s, j->sp, j->d, j->dp, j->w, j->h);
    }
}
static DWORD WINAPI h4c_thread(LPVOID p)
{
    (void)p;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    for (;;) {
        WaitForSingleObject(s_h4c_sem, INFINITE);
        h4c_drain();
        InterlockedIncrement(&s_h4c_done);   /* exactly one per consumed semaphore count */
    }
    return 0;
}
static int h4c_threads(void)
{
    static int v = -1, ab = -1;
    if (v < 0) { const char* e = getenv("BLIT_CONV_MT"); v = e ? atoi(e) : 0; if (v < 0) v = 0; if (v > 7) v = 7;
                 fprintf(stderr, "[HP4B] BLIT_CONV_MT=%d (deferred G-buffer rotate helper threads)%c", v, 10); }
    const int on = nl_ab_flip(v > 0, &ab, "BLITCONV");
    return on ? (v > 0 ? v : 3) : 0;
}
static int h4c_verify_on(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("BLIT_CONV_MT_VERIFY"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v;
}
static int h4c_ensure(int n)
{
    if (s_h4c_nthr >= n) return n;
    if (!s_h4c_sem) s_h4c_sem = CreateSemaphoreA(NULL, 0, 64, NULL);
    if (!s_h4c_sem) return 0;
    while (s_h4c_nthr < n) {
        HANDLE h = CreateThread(NULL, 256u << 10, h4c_thread, NULL, 0, NULL);
        if (!h) break;
        CloseHandle(h); s_h4c_nthr++;
    }
    fprintf(stderr, "[HP4B] BLIT_CONV_MT: %d helper threads running%c", s_h4c_nthr, 10);
    return s_h4c_nthr < n ? s_h4c_nthr : n;
}
/* Converts the strips (already resolved to host pointers) with nthr helpers + the caller. Caller holds whatever
 * locks keep the source/destination alive; returns only after every helper has left the job array. */
static void h4c_run(const H4ConvJob* strips, int ns, int nthr)
{
    nthr = h4c_ensure(nthr);
    /* chunks: ~4 per participating thread per strip, >= 16 rows */
    int nj = 0;
    for (int i = 0; i < ns; i++) {
        const H4ConvJob* s = &strips[i];
        u32 rows = s->h / (u32)(4 * (nthr + 1)); if (rows < 16) rows = 16;
        for (u32 y = 0; y < s->h && nj < H4C_MAXJOBS; y += rows) {
            H4ConvJob* j = &s_h4c_job[nj++];
            const u32 hh = (s->h - y < rows) ? s->h - y : rows;
            j->s = s->s + (size_t)y * s->sp; j->sp = s->sp; j->d = s->d + (size_t)y * s->dp; j->dp = s->dp; j->w = s->w; j->h = hh;
            if (nj == H4C_MAXJOBS && y + hh < s->h) {   /* never truncate: last job takes the remainder */
                j->h = s->h - y;
            }
        }
    }
    if (nthr <= 0 || nj <= 1) {
        for (int i = 0; i < nj; i++) rsx_blit_rgba_to_argb_rows(s_h4c_job[i].s, s_h4c_job[i].sp, s_h4c_job[i].d, s_h4c_job[i].dp, s_h4c_job[i].w, s_h4c_job[i].h);
        return;
    }
    s_h4c_njob = nj;
    InterlockedExchange(&s_h4c_done, 0);
    InterlockedExchange(&s_h4c_next, 0);
    ReleaseSemaphore(s_h4c_sem, nthr, NULL);
    h4c_drain();
    for (unsigned k = 0; InterlockedCompareExchange(&s_h4c_done, 0, 0) < nthr; k++) {
        if (k < 4096) _mm_pause(); else SwitchToThread();
    }
    s_h4c_njob = 0;
}

/* ---- TEX_SNAP_UPLOAD: snapshot arena = CPU-cached D3D12 buffer the GPU copies textures from ---- */
static ID3D12Resource* s_h4_snap_res[2];
static u8* s_h4_snap_map[2];
static size_t s_h4_snap_cap = 0;
static int s_h4_snap_state = -1;     /* -1 untried, 0 failed/off, 1 ready */
static int s_h4_snap_par_on[2];       /* decided at each hand-off */
static int h4_snapup_on(void)
{
    static int v = -1, ab = -1;
    if (v < 0) { const char* e = getenv("TEX_SNAP_UPLOAD"); v = (e && *e && *e != '0') ? 1 : 0;
                 fprintf(stderr, "[HP4B] TEX_SNAP_UPLOAD=%d (GPU copies snapshot textures straight from the snapshot arena)%c", v, 10);
                 if (getenv("TEX_SAVE") || getenv("RTT_DUMP") || getenv("TEX_HILITE") || getenv("TEX_MARKEMPTY") ||
                     getenv("TEX_OFF_BIAS") || getenv("TEX_REMAP") || getenv("TEX_SRCDBG") || getenv("MOVIE_FIND") || getenv("TEX_NOCACHE")) {
                     if (v) fprintf(stderr, "[HP4B] TEX_SNAP_UPLOAD forced off: a texture diagnostic env var is set%c", 10);
                     v = 0; ab = 0; } }
    if (v == 0 && ab == 0) return 0;
    return nl_ab_flip(v, &ab, "SNAPUP");
}
/* FIFO thread, at the hand-off (render thread idle; arena[par]'s previous GPU use finished: the render thread's
 * last replay started with wait_for_gpu()). */
static int h4_snapup_ready(void)
{
    if (s_h4_snap_state >= 0) return s_h4_snap_state;
    s_h4_snap_state = 0;
    if (!s_d3d.device) return 0;
    long mb = 32; { const char* e = getenv("TEX_SNAP_UPLOAD_MB"); if (e) mb = atol(e); if (mb < 4) mb = 4; if (mb > 256) mb = 256; }
    D3D12_HEAP_PROPERTIES hp; memset(&hp, 0, sizeof hp);
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    hp.CreationNodeMask = 1; hp.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC bd; memset(&bd, 0, sizeof bd);
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = (u64)mb << 20;
    bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    HRESULT hr = S_OK;
    for (int p = 0; p < 2; p++) {
        hr = s_d3d.device->lpVtbl->CreateCommittedResource(s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &bd,
                 D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource, (void**)&s_h4_snap_res[p]);
        if (FAILED(hr)) { s_h4_snap_res[p] = NULL; break; }
        D3D12_RANGE nr = {0, 0}; void* m = NULL;
        hr = s_h4_snap_res[p]->lpVtbl->Map(s_h4_snap_res[p], 0, &nr, &m);
        if (FAILED(hr) || !m) { if (SUCCEEDED(hr)) hr = E_FAIL; break; }
        s_h4_snap_map[p] = (u8*)m;
    }
    if (FAILED(hr)) {
        for (int p = 0; p < 2; p++) { if (s_h4_snap_res[p]) s_h4_snap_res[p]->lpVtbl->Release(s_h4_snap_res[p]); s_h4_snap_res[p] = NULL; s_h4_snap_map[p] = NULL; }
        fprintf(stderr, "[HP4B] TEX_SNAP_UPLOAD: custom write-back heap buffer FAILED hr=0x%08lX -> off%c", (long)hr, 10);
        return 0;
    }
    s_h4_snap_cap = (size_t)bd.Width; s_h4_snap_state = 1;
    fprintf(stderr, "[HP4B] TEX_SNAP_UPLOAD: 2 x %ld MB CPU-cached snapshot arenas ready%c", mb, 10);
    return 1;
}

/* ---- HP4B_HEAP_PROBE: OpenExistingHeapFromAddress on vm_base (log only) ---- */
static const GUID h4_IID_ID3D12Device3 = { 0x81dadc15, 0x2bad, 0x4392, { 0x93, 0xc5, 0x10, 0x13, 0x45, 0xc4, 0xaa, 0x98 } };
static void h4_heap_probe(void)
{
    static int done = 0;
    if (done) return;
    done = 1;
    { const char* e = getenv("HP4B_HEAP_PROBE"); if (!(e && *e == '1')) return; }
    extern uint8_t* vm_base;
    if (!s_d3d.device || !vm_base) { fprintf(stderr, "[HP4B] heap probe: no device/vm_base%c", 10); return; }
    MEMORY_BASIC_INFORMATION mbi; memset(&mbi, 0, sizeof mbi);
    VirtualQuery(vm_base, &mbi, sizeof mbi);
    fprintf(stderr, "[HP4B] heap probe: vm_base=%p AllocationBase=%p region=%llu MB state=0x%lX protect=0x%lX type=0x%lX%c",
            (void*)vm_base, mbi.AllocationBase, (unsigned long long)(mbi.RegionSize >> 20), (unsigned long)mbi.State,
            (unsigned long)mbi.Protect, (unsigned long)mbi.Type, 10);
    ID3D12Device3* d3 = NULL;
    HRESULT hr = s_d3d.device->lpVtbl->QueryInterface(s_d3d.device, &h4_IID_ID3D12Device3, (void**)&d3);
    if (FAILED(hr) || !d3) { fprintf(stderr, "[HP4B] heap probe: no ID3D12Device3 hr=0x%08lX%c", (long)hr, 10); return; }
    /* control: a private 1 MB VirtualAlloc must work if the feature exists at all */
    void* priv = VirtualAlloc(NULL, 1u << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (priv) {
        ID3D12Heap* h = NULL;
        hr = d3->lpVtbl->OpenExistingHeapFromAddress(d3, priv, &IID_ID3D12Heap, (void**)&h);
        D3D12_HEAP_DESC hd; memset(&hd, 0, sizeof hd);
        if (SUCCEEDED(hr) && h) { h->lpVtbl->GetDesc(h, &hd); }
        fprintf(stderr, "[HP4B] heap probe: private 1 MB VirtualAlloc -> hr=0x%08lX heap size %llu%c", (long)hr,
                (unsigned long long)hd.SizeInBytes, 10);
        if (h) h->lpVtbl->Release(h);
    }
    {   /* the guest window itself (the main-memory commit is the first region of the reservation) */
        ID3D12Heap* h = NULL;
        hr = d3->lpVtbl->OpenExistingHeapFromAddress(d3, vm_base, &IID_ID3D12Heap, (void**)&h);
        D3D12_HEAP_DESC hd; memset(&hd, 0, sizeof hd);
        if (SUCCEEDED(hr) && h) { h->lpVtbl->GetDesc(h, &hd); }
        fprintf(stderr, "[HP4B] heap probe: vm_base -> hr=0x%08lX heap size %llu MB (%s)%c", (long)hr,
                (unsigned long long)(hd.SizeInBytes >> 20), SUCCEEDED(hr) ? "WORKS: GUESTHEAP feasible" : "refused", 10);
        if (h) h->lpVtbl->Release(h);
    }
    if (priv) VirtualFree(priv, 0, MEM_RELEASE);
    d3->lpVtbl->Release(d3);
    fflush(stderr);
}
/* ===== end HP4B ===== */
