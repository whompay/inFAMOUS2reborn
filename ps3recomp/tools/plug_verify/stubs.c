/* v134 harness runtime stubs: just enough of the SPU runtime to run (a) the interpreter step loop and (b) the
 * relocatable lifted plugin code on a dumped LS + register state, stopping at the first exit from the blob. */
#include "spu_context.h"
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
SPU_THREAD_LOCAL void (*g_spu_trampoline_fn)(spu_context*);
SPU_THREAD_LOCAL uint32_t g_spu_pch[8];
SPU_THREAD_LOCAL unsigned g_spu_pch_n;
SPU_THREAD_LOCAL uint32_t g_spu_recent[32];
SPU_THREAD_LOCAL uint32_t g_spu_recent_n;
volatile int g_yz_lockstep_on = 0;
int g_spu_ls_watch_n = 0;
volatile int g_ab_mask = 0;
volatile int g_abx = 0x10;
uint8_t g_spu_fn_bitmap[0x40000 / 32];
volatile uint32_t g_spu_fn_reg_gen = 1;
#include "spu_laneprof.h"
SPU_THREAD_LOCAL spu_laneprof_tls g_spu_lp;
void yz_lockstep_tick(spu_context* c) { (void)c; }
void spu_ls_watch_slow(uint32_t a, int w, const uint8_t* p, uint32_t pc, uint32_t lr) { (void)a;(void)w;(void)p;(void)pc;(void)lr; }
void spu_tlc_slow(spu_context* c, void* f) { (void)c; (void)f; }
void spu_depth_guard(spu_context* c) { (void)c; }
int spu_tailret_enabled(void) { g_spu_tailret = 0; return 0; }
void spu_img_restore(spu_context* c, int32_t s) { (void)c; (void)s; }
void spu_check_stack_reset_slow(spu_context* c) { (void)c; }

/* harness control */
jmp_buf g_hv_jb; int g_hv_kind; uint32_t g_hv_pc; uint32_t g_hv_base, g_hv_end;
enum { HV_OUT = 1, HV_CHAN = 2, HV_STOP = 3, HV_MID = 4, HV_UNSUP = 5 };
void hv_exit(int kind, uint32_t pc) { g_hv_kind = kind; g_hv_pc = pc; longjmp(g_hv_jb, 1); }
typedef void (*spu_plug_fn)(spu_context*);
spu_plug_fn spu_plug_lookup(spu_context* ctx, uint32_t pc);
void spu_indirect_branch(spu_context* ctx)
{
    uint32_t pc = ctx->pc & SPU_LS_MASK; ctx->pc = pc;
    if (pc < g_hv_base || pc >= g_hv_end) hv_exit(HV_OUT, pc);
    spu_plug_fn f = spu_plug_lookup(ctx, pc);
    if (!f) hv_exit(HV_MID, pc);
    f(ctx);
}
void spu_indirect_branch_mt(spu_context* ctx) { spu_indirect_branch(ctx); }
void spu_drain_call(spu_context* ctx, uint32_t ret)
{
    while (g_spu_trampoline_fn) { void (*tf)(spu_context*) = g_spu_trampoline_fn; g_spu_trampoline_fn = 0; tf(ctx); }
    if ((ctx->pc & SPU_LS_MASK) != (ret & SPU_LS_MASK)) {
        /* returned elsewhere: in the real runtime the dispatch restarts at ctx->pc */
        fprintf(stderr, "  [drain] returned to 0x%05X expected 0x%05X\n", ctx->pc & SPU_LS_MASK, ret & SPU_LS_MASK);
        hv_exit(HV_OUT, ctx->pc & SPU_LS_MASK);
    }
}
u128 spu_rdch(spu_context* ctx, uint32_t ch) { (void)ctx; hv_exit(HV_CHAN, ch); return (u128){0}; }
void spu_wrch(spu_context* ctx, uint32_t ch, u128 v) { (void)ctx; (void)v; hv_exit(HV_CHAN, 0x100 | ch); }
uint32_t spu_rchcnt(spu_context* ctx, uint32_t ch) { (void)ctx; hv_exit(HV_CHAN, 0x200 | ch); return 0; }
void spu_stop(spu_context* ctx) { hv_exit(HV_STOP, ctx->stop_code); }
void spu_halt(spu_context* ctx) { (void)ctx; hv_exit(HV_STOP, 0xFFFF); }
void spu_unsupported(uint32_t pc, const char* mn) { (void)mn; hv_exit(HV_UNSUP, pc); }
void (*spu_take_interrupt(spu_context* ctx, void (*tf)(spu_context*)))(spu_context*) { (void)ctx; return tf; }

