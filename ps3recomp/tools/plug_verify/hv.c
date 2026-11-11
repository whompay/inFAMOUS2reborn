/* v134 harness: interpreter vs relocatable lifted plugin code from the same dumped state. */
#define spu_rdch hv_unused_rdch_decl
#undef spu_rdch
#include "spu_interp.c"
#include <setjmp.h>
extern jmp_buf g_hv_jb; extern int g_hv_kind; extern uint32_t g_hv_pc, g_hv_base, g_hv_end;
typedef void (*spu_plug_fn)(spu_context*);
spu_plug_fn spu_plug_lookup(spu_context* ctx, uint32_t pc);
extern uint32_t g_spu_frag_delta;
spu_lifted_fn spu_lifted_lookup(const spu_context* ctx, uint32_t lsa) { (void)ctx; (void)lsa; return 0; }
void spu_spurs_taskset_syscall(spu_context* c) { (void)c; }
void spu_wws_jobapi_trace(spu_context* c) { (void)c; }
static uint8_t s_ls1[SPU_LS_SIZE], s_ls2[SPU_LS_SIZE], s_ls0[SPU_LS_SIZE];
static int is_chan(uint32_t w) { uint32_t op11 = w >> 21; return op11 == 0x00D || op11 == 0x10D || op11 == 0x00F || (w >> 21) == 0; }
int main(int argc, char** argv)
{
    int fails = 0, ok = 0, skip = 0;
    for (int ai = 1; ai < argc; ai += 3) {
        const char* fn = argv[ai]; uint32_t base = strtoul(argv[ai+1], 0, 16), size = strtoul(argv[ai+2], 0, 16);
        const char* p = strrchr(fn, '_'); p -= 5; uint32_t pc0 = strtoul(p, 0, 16);
        FILE* f = fopen(fn, "rb"); if (!f) { perror(fn); continue; }
        static spu_context c0, c1, c2;
        memset(&c0, 0, sizeof c0);
        fread(s_ls0, 1, SPU_LS_SIZE, f); fread(c0.gpr, 1, sizeof c0.gpr, f); fclose(f);
        c0.ls = s_ls0; c0.image_id = 1; c0.pc = pc0; c0.host_depth = 1; c0.int_enable = 0;
        g_hv_base = base + 0x30; g_hv_end = base + size;
        /* interpreter */
        c1 = c0; memcpy(s_ls1, s_ls0, SPU_LS_SIZE); c1.ls = s_ls1;
        int k1 = 0; uint32_t x1 = 0; uint64_t steps = 0;
        for (;;) {
            uint32_t pc = c1.pc & SPU_LS_MASK;
            if (pc < g_hv_base || pc >= g_hv_end) { k1 = 1; x1 = pc; break; }
            uint32_t w = ((uint32_t)s_ls1[pc] << 24) | ((uint32_t)s_ls1[pc+1] << 16) | ((uint32_t)s_ls1[pc+2] << 8) | s_ls1[pc+3];
            if (is_chan(w)) { k1 = (w >> 21) == 0 ? 3 : 2; x1 = (w >> 21) == 0 ? (w & 0x3FFF) : 0; break; }
            if (spu_step(&c1)) { k1 = 3; x1 = c1.stop_code; break; }
            if (++steps > 200000000ull) { k1 = 9; break; }
        }
        /* lifted */
        c2 = c0; memcpy(s_ls2, s_ls0, SPU_LS_SIZE); c2.ls = s_ls2;
        spu_plug_fn lf = spu_plug_lookup(&c2, pc0);
        if (!lf) { printf("SKIP %s pc %05X: not a lifted entry\n", fn, pc0); skip++; continue; }
        int k2; uint32_t x2;
        if (!setjmp(g_hv_jb)) {
            lf(&c2);
            for (;;) {
                while (g_spu_trampoline_fn) { void (*tf)(spu_context*) = g_spu_trampoline_fn; g_spu_trampoline_fn = 0; tf(&c2); }
                uint32_t pc = c2.pc & SPU_LS_MASK;
                if (pc < g_hv_base || pc >= g_hv_end) break;
                spu_plug_fn nf = spu_plug_lookup(&c2, pc);   /* returned into the blob without a live frame: dispatch */
                if (!nf) { g_hv_kind = 4; g_hv_pc = pc; longjmp(g_hv_jb, 1); }
                nf(&c2);
            }
            k2 = 1; x2 = c2.pc & SPU_LS_MASK;
        } else { k2 = g_hv_kind; x2 = g_hv_pc; }
        if (k2 == 4) { printf("MID  %s: lifted left to unlifted blob pc %05X (off %X); interp exit %d %05X after %llu steps\n", fn, x2, x2 - base, k1, x1, (unsigned long long)steps); skip++; continue; }
        int greg = memcmp(c1.gpr, c2.gpr, sizeof c1.gpr) != 0;
        int lsd = memcmp(s_ls1, s_ls2, SPU_LS_SIZE) != 0;
        int exd = (k1 != k2) || (k1 == 1 && x1 != x2);
        if (greg || lsd || exd) {
            fails++;
            printf("FAIL %s base %05X: interp exit %d %05X (%llu steps) | lifted exit %d %05X | regs %s ls %s\n", fn, base, k1, x1, (unsigned long long)steps, k2, x2, greg ? "DIFF" : "ok", lsd ? "DIFF" : "ok");
            for (int r = 0; r < 128 && greg; r++) if (memcmp(&c1.gpr[r], &c2.gpr[r], 16))
                printf("   r%d interp %08X %08X %08X %08X lifted %08X %08X %08X %08X\n", r, c1.gpr[r]._u32[0], c1.gpr[r]._u32[1], c1.gpr[r]._u32[2], c1.gpr[r]._u32[3], c2.gpr[r]._u32[0], c2.gpr[r]._u32[1], c2.gpr[r]._u32[2], c2.gpr[r]._u32[3]);
            if (lsd) { int n = 0; for (uint32_t a = 0; a < SPU_LS_SIZE && n < 6; a += 16) if (memcmp(s_ls1 + a, s_ls2 + a, 16)) { printf("   LS %05X differs\n", a); n++; } }
        } else { ok++; printf("ok   %s exit %d %05X (%llu steps)\n", fn, k1, x1, (unsigned long long)steps); }
    }
    printf("SUMMARY ok %d fail %d skip %d\n", ok, fails, skip);
    return 0;
}
