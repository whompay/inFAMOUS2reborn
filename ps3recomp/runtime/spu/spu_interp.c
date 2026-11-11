/* spu_interp.c — SPU interpreter core. See spu_interp.h.
 *
 * Decode tables are generated from the validated Python decoder
 * (tools/gen_spu_interp.py -> spu_interp_tables.inc); execute calls the SAME
 * spu_<mnemonic> helpers the lifter emits, so interpreted and lifted code are
 * bit-identical. Uncommon opcodes trap loudly (never silently wrong).
 */
#include "../../libs/video/rsx_getenv_cache.h"   /* cached getenv(): SPU hot paths called getenv per PM run (2026-10-04) */
#include <xmmintrin.h>
#include "spu_interp.h"
#include "spu_helpers.h"
/* v98f: the whole-register SSE/AVX2 forms of the integer helpers (v95, tested bit-identical to
 * spu_helpers.h by tests/test_spu_helpers_vec.c). The LS accessors stay the spu_context.h ones
 * (they keep the SPU_LS_WATCH / LOWREAD / SMC_WATCH probes). SPU_INTERP_NO_VEC builds without. */
#if !defined(SPU_INTERP_NO_VEC)
#include "spu_helpers_vec.h"
#undef spu_ls_read128
#undef spu_ls_write128
#endif
#include <stdio.h>
#include <stdlib.h>

/* Channel ABI (runtime/spu/spu_channels.c); the lifter emits these same protos
 * into each generated spu_recomp.h — declared here to stay header-independent. */
u128 spu_rdch(spu_context* ctx, uint32_t channel);
void spu_wrch(spu_context* ctx, uint32_t channel, u128 value);
uint32_t spu_rchcnt(spu_context* ctx, uint32_t channel);

#include "spu_interp_tables.inc"   /* spu_op enum, spu_op_name[], spu_dec_* */
#include "spu_laneprof.h"   /* v96a */

/* Work descriptor for the next SYS_SPU_THREAD_STOP_RECEIVE_EVENT service, set
 * by the event-port send that wakes a sim SPU. Dispatch is synchronous on the
 * sending PPU host thread. The raw SPU also has an asynchronous startup
 * instance, so a process-global slot races and can be consumed by the wrong
 * instance. Keep the handoff on the dispatching host thread instead. */
#if defined(_MSC_VER)
#  define SPU_INTERP_TLS __declspec(thread)
#else
#  define SPU_INTERP_TLS __thread
#endif
/* v52: approximate SPU extended-range floats (spu_helpers.h spu_xfv). SPU_XFLOAT=0 = off. */
int g_spu_xfloat = 1;
/* v131: spu_helpers.h spu__xfq clamp limits (SSE4.1 path); must follow g_spu_xfloat. */
uint32_t g_spu_xflim[8] = { 0xFF7FFFFFu, 0xFF7FFFFFu, 0xFF7FFFFFu, 0xFF7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu };
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor)) static void spu_xfloat_env(void)
{
    const char* e = getenv("SPU_XFLOAT");
    if (e && *e == '0') { g_spu_xfloat = 0; for (int i = 0; i < 4; i++) { g_spu_xflim[i] = 0xFFFFFFFFu; g_spu_xflim[4 + i] = 0x7FFFFFFFu; } }
    fprintf(stderr, "[SPU] v52 approximate xfloat (exponent-255 operands = +-FLT_MAX) %s (SPU_XFLOAT)\n", g_spu_xfloat ? "ON" : "OFF");
}
#endif
static SPU_INTERP_TLS uint32_t s_spu_pending_evt[3];
static SPU_INTERP_TLS int      s_spu_pending_evt_valid;

void spu_interp_set_pending_event(uint32_t data1, uint32_t data2, uint32_t data3)
{
    s_spu_pending_evt[0] = data1;
    s_spu_pending_evt[1] = data2;
    s_spu_pending_evt[2] = data3;
    s_spu_pending_evt_valid = 1;
}

void spu_interp_clear_pending_event(void)
{
    s_spu_pending_evt_valid = 0;
}

/* ---- decode: 32-bit insn -> fields (mirrors spu_disasm.spu_decode order) ---- */
typedef struct {
    spu_op   op;
    uint8_t  rt, ra, rb, rc, ch;
    int32_t  imm;     /* sign/zero-extended immediate or shift, per format */
    uint32_t tgt;     /* absolute LS branch target (br-family) */
} spu_ins;

static int32_t sx(uint32_t v, int bits) {
    uint32_t m = 1u << (bits - 1);
    return (int32_t)((v ^ m) - m);
}

static void spu_decode1(uint32_t insn, uint32_t pc, spu_ins* d) {
    uint32_t op4 = (insn >> 28) & 0xF,  op7 = (insn >> 25) & 0x7F;
    uint32_t op8 = (insn >> 24) & 0xFF, op9 = (insn >> 23) & 0x1FF;
    uint32_t op10 = (insn >> 22) & 0x3FF, op11 = (insn >> 21) & 0x7FF;
    d->rt = insn & 0x7F; d->ra = (insn >> 7) & 0x7F;
    d->rb = (insn >> 14) & 0x7F; d->rc = (insn >> 21) & 0x7F;
    d->ch = (insn >> 7) & 0x7F; d->imm = 0; d->tgt = 0;
    int32_t i10 = sx((insn >> 14) & 0x3FF, 10);
    int32_t i16 = sx((insn >> 7) & 0xFFFF, 16);
    uint32_t i18 = (insn >> 7) & 0x3FFFF;

    /* RRR (4) */
    if ((d->op = spu_dec_rrr(op4)) != SPU_word) return;
    /* RI18 (7) */
    if ((d->op = spu_dec_ri18(op7)) != SPU_word) {
        d->imm = (d->op == SPU_ila) ? (int32_t)i18 : (int32_t)(i18 & 0xFFFF);
        return;
    }
    /* RI16 (9) — before RI10 (shared top-8-bit opcodes) */
    if ((d->op = spu_dec_ri16(op9)) != SPU_word) {
        switch (d->op) {
        case SPU_lqa: case SPU_stqa: d->tgt = ((uint32_t)(i16 & 0xFFFF) << 2) & 0x3FFF0; break;
        case SPU_lqr: case SPU_stqr: d->tgt = ((uint32_t)(i16 * 4) + pc) & 0x3FFF0; break;
        case SPU_br: case SPU_brsl: case SPU_brz: case SPU_brnz:
        case SPU_brhz: case SPU_brhnz: d->tgt = ((uint32_t)(i16 * 4) + pc) & 0x3FFFF; break;
        case SPU_bra: case SPU_brasl: d->tgt = ((uint32_t)(i16 * 4)) & 0x3FFFF; break;
        case SPU_il: d->imm = i16; break;
        default: d->imm = (int32_t)(i16 & 0xFFFF); break; /* ilh/ilhu/iohl */
        }
        return;
    }
    /* channel ops (op11) — before RI10 (wrch/shli op8 clash) */
    if (op11 == 0x00D) { d->op = SPU_rdch;   return; }
    if (op11 == 0x10D) { d->op = SPU_wrch;   return; }
    if (op11 == 0x00F) { d->op = SPU_rchcnt; return; }
    /* RI10 (8) */
    if ((d->op = spu_dec_ri10(op8)) != SPU_word) {
        d->imm = (d->op == SPU_lqd || d->op == SPU_stqd) ? (i10 << 4) : i10;
        return;
    }
    /* RI8 (10) float<->int conversions */
    if ((d->op = spu_dec_ri8(op10)) != SPU_word) { d->imm = (insn >> 14) & 0xFF; return; }
    /* RR (11) */
    if ((d->op = spu_dec_rr(op11)) != SPU_word) {
        if (d->op == SPU_stop) d->imm = insn & 0x3FFF;
        return;
    }
    /* RI7 (11) rotate/shift/gen-control immediates */
    if ((d->op = spu_dec_ri7(op11)) != SPU_word) {
        d->imm = spu_ri7_unsigned(d->op) ? d->rb : sx(d->rb, 7);
        return;
    }
    /* LSX (redundant with RR's lqx/stqx, kept for order-parity) */
    if ((d->op = spu_dec_lsx(op11)) != SPU_word) return;
    d->op = SPU_word;
}

/* spu_lifted_lookup() is defined by spu_fn_registry.c (the single owner of the
 * lifted-function table + eviction). Declared in spu_interp.h. */

/* ---- execute one instruction; returns 1 if the SPU stopped ---- */
#define A   (ctx->gpr[d.ra])
#define B   (ctx->gpr[d.rb])
#define T   (ctx->gpr[d.rt])          /* also the 3rd source in RRR */
#define DST (ctx->gpr[d.rt])          /* default destination */
#define DSTC (ctx->gpr[d.rc])         /* RRR destination */
#define I   (d.imm)
#define PREF(r) ((r)._u32[0])

/* v98f: per-thread decode cache for the fast interpreter loop. Keyed by (pc, instruction word):
 * a hit reuses the decode of the very same word at the very same pc, so it is exact even when
 * the code in local store changes (runtime-generated / per-batch code fragments). */
#define SPU_DC_N 4096
typedef struct { uint32_t pc, insn; spu_ins d; uint32_t gen, flags; } spu_dc_ent;   /* v98f stage 2: + gen/flags */
static SPU_INTERP_TLS spu_dc_ent t_spu_dc[SPU_DC_N];
/* v98f differential test (SPU_INTERP_DIFF): dry runs stop BEFORE any instruction with an effect
 * outside gpr/LS (channel ops, stop, iret, interrupt branches, conditional halts). */
static SPU_INTERP_TLS int t_spu_dry, t_spu_dry_abort;
#define SPU_DRY_GUARD() do { if (__builtin_expect(t_spu_dry, 0)) { t_spu_dry_abort = 1; ctx->pc = pc; ctx->stop_code = 0; ctx->status = SPU_STATUS_STOPPED_BY_HALT; return 1; } } while (0)

/* v98f stage 2: execute one already-decoded instruction (shared by both loops). */
static inline __attribute__((always_inline)) int spu_step_impl(spu_context* ctx, const uint32_t pc, const uint32_t insn, const spu_ins d, const int cached) {
    uint32_t next = pc + 4;

    switch (d.op) {
    /* immediates / loads */
    case SPU_il: DST = spu_splat_u32((uint32_t)(int32_t)I); break;
    /* ilh replicates the 16-bit immediate into all EIGHT halfwords. It used to
     * share il's word splat (0x0000XXXX), zeroing every odd halfword lane of
     * the constant -- the interpreted Bink decoder (inFamous) then left every
     * other 32-px column band of its frames unwritten. 2026-10-03. */
    case SPU_ilh: DST = spu_ilh((uint16_t)I); break;
    case SPU_ilhu: DST = spu_splat_u32((uint32_t)I << 16); break;
    case SPU_ila:  DST = spu_splat_u32((uint32_t)I & 0x3FFFF); break;
    case SPU_iohl: DST = spu_ori(DST, (int32_t)((uint32_t)I & 0xFFFF)); break;
    case SPU_fsmbi: DST = spu_fsmbi(I); break;
    case SPU_lqd: case SPU_lqx: case SPU_lqa: case SPU_lqr: {
        uint32_t a = (d.op==SPU_lqd) ? PREF(A)+ (uint32_t)I
                   : (d.op==SPU_lqx) ? PREF(A)+PREF(B) : d.tgt;
        DST = spu_ls_read128(ctx, a); break; }
    case SPU_stqd: case SPU_stqx: case SPU_stqa: case SPU_stqr: {
        uint32_t a = (d.op==SPU_stqd) ? PREF(A)+ (uint32_t)I
                   : (d.op==SPU_stqx) ? PREF(A)+PREF(B) : d.tgt;
        spu_ls_write128(ctx, a, DST); break; }
    /* integer arithmetic */
    case SPU_a:  DST = spu_a(A,B); break;
    case SPU_ai: DST = spu_ai(A,I); break;
    case SPU_sf: DST = spu_sf(A,B); break;
    case SPU_sfi:DST = spu_sfi(A,I); break;
    case SPU_ah: DST = spu_ah(A,B); break;
    case SPU_ahi:DST = spu_ahi(A,I); break;
    case SPU_sfh:DST = spu_sfh(A,B); break;
    case SPU_sfhi:DST = spu_sfhi(A,I); break;
    case SPU_addx: DST = spu_addx(A,B,DST); break;
    case SPU_sfx:  DST = spu_sfx(A,B,DST); break;
    case SPU_cg:   DST = spu_cg(A,B); break;
    case SPU_cgx:  DST = spu_cgx(A,B,DST); break;
    case SPU_mpyi: DST = spu_mpyi(A,I); break;
    /* logical */
    case SPU_and: DST = spu_and(A,B); break;
    case SPU_or:  DST = spu_or(A,B); break;
    case SPU_xor: DST = spu_xor(A,B); break;
    case SPU_nand:DST = spu_nand(A,B); break;
    case SPU_nor: DST = spu_nor(A,B); break;
    case SPU_andc:DST = spu_andc(A,B); break;
    case SPU_orc: DST = spu_orc(A,B); break;
    case SPU_eqv: DST = spu_eqv(A,B); break;
    case SPU_andi:DST = spu_andi(A,I); break;
    case SPU_ori: DST = spu_ori(A,I); break;
    case SPU_xori:DST = spu_xori(A,I); break;
    case SPU_andbi:DST = spu_andbi(A,I); break;
    case SPU_andhi:DST = spu_andhi(A,I); break;
    case SPU_orhi: DST = spu_orhi(A,I); break;
    case SPU_orbi: DST = spu_orbi(A,I); break;
    case SPU_xorhi:DST = spu_xorhi(A,I); break;
    case SPU_xorbi:DST = spu_xorbi(A,I); break;
    case SPU_orx:  DST = spu_orx(A); break;
    /* compares */
    case SPU_ceq: DST = spu_ceq(A,B); break;
    case SPU_ceqh:DST = spu_ceqh(A,B); break;
    case SPU_ceqb:DST = spu_ceqb(A,B); break;
    case SPU_ceqi:DST = spu_ceqi(A,I); break;
    case SPU_cgt: DST = spu_cgt(A,B); break;
    case SPU_cgth:DST = spu_cgth(A,B); break;
    case SPU_cgtb:DST = spu_cgtb(A,B); break;
    case SPU_cgti:DST = spu_cgti(A,I); break;
    case SPU_clgt: DST = spu_clgt(A,B); break;
    case SPU_clgth:DST = spu_clgth(A,B); break;
    case SPU_clgtb:DST = spu_clgtb(A,B); break;
    case SPU_clgti:DST = spu_clgti(A,I); break;
    /* shifts / rotates */
    case SPU_shl:  DST = spu_shl(A,B); break;
    case SPU_shli: DST = spu_shli(A,I); break;
    case SPU_shlh: DST = spu_shlh(A,B); break;
    case SPU_shlhi:DST = spu_shlhi(A,I); break;
    case SPU_rot:  DST = spu_rot(A,B); break;
    case SPU_roti: DST = spu_roti(A,I); break;
    case SPU_rotm: DST = spu_rotm(A,B); break;
    case SPU_rotmi:DST = spu_rotmi(A,I); break;
    case SPU_rotma:DST = spu_rotma(A,B); break;
    case SPU_rotmai:DST = spu_rotmai(A,I); break;
    case SPU_roth: DST = spu_roth(A,B); break;
    case SPU_rothi:DST = spu_rothi(A,I); break;
    case SPU_shlqbi: DST = spu_shlqbi(A,B); break;
    case SPU_shlqbii:DST = spu_shlqbii(A,I); break;
    case SPU_shlqby: DST = spu_shlqby(A,B); break;
    case SPU_shlqbyi:DST = spu_shlqbyi(A,I); break;
    case SPU_rotqbi: DST = spu_rotqbi(A,B); break;
    case SPU_rotqbii:DST = spu_rotqbii(A,I); break;
    case SPU_rotqby: DST = spu_rotqby(A,B); break;
    case SPU_rotqbyi:DST = spu_rotqbyi(A,I); break;
    case SPU_rotqmby: DST = spu_rotqmby(A,B); break;
    case SPU_rotqmbyi:DST = spu_rotqmbyi(A,I); break;
    /* select / shuffle / masks / bitcount (RRR: dest=rc, srcs=ra,rb,rt) */
    case SPU_selb:  DSTC = spu_selb(A,B,T); break;
    case SPU_shufb: DSTC = spu_shufb(A,B,T); break;
    case SPU_fsm:  DST = spu_fsm(A); break;
    case SPU_fsmb: DST = spu_fsmb(A); break;
    case SPU_fsmh: DST = spu_fsmh(A); break;
    case SPU_gb:   DST = spu_gb(A); break;
    case SPU_gbb:  DST = spu_gbb(A); break;
    case SPU_gbh:  DST = spu_gbh(A); break;
    case SPU_clz:  DST = spu_clz(A); break;
    case SPU_cntb: DST = spu_cntb(A); break;
    /* gen-controls for insertion */
    case SPU_cbd: DST = spu_cbd(A,I); break;
    case SPU_chd: DST = spu_chd(A,I); break;
    case SPU_cwd: DST = spu_cwd(A,I); break;
    case SPU_cdd: DST = spu_cdd(A,I); break;
    case SPU_cbx: DST = spu_cbx(A,B); break;
    case SPU_chx: DST = spu_chx(A,B); break;
    case SPU_cwx: DST = spu_cwx(A,B); break;
    case SPU_cdx: DST = spu_cdx(A,B); break;
    /* multiply */
    case SPU_mpy:  DST = spu_mpy(A,B); break;
    case SPU_mpyu: DST = spu_mpyu(A,B); break;
    case SPU_mpyui:DST = spu_mpyui(A,I); break;
    case SPU_mpyh: DST = spu_mpyh(A,B); break;
    case SPU_mpya: DSTC = spu_mpya(A,B,T); break;
    /* sign-extend */
    case SPU_xsbh: DST = spu_xsbh(A); break;
    case SPU_xshw: DST = spu_xshw(A); break;
    case SPU_xswd: DST = spu_xswd(A); break;
    /* float */
    case SPU_fa: DST = spu_fa(A,B); break;
    case SPU_fs: DST = spu_fs(A,B); break;
    case SPU_fm: DST = spu_fm(A,B); break;
    case SPU_fcgt: DST = spu_fcgt(A,B); break;
    case SPU_fceq: DST = spu_fceq(A,B); break;
    case SPU_cflts: DST = spu_cflts(A,I); break;
    case SPU_cfltu: DST = spu_cfltu(A,I); break;
    case SPU_csflt: DST = spu_csflt(A,I); break;
    case SPU_cuflt: DST = spu_cuflt(A,I); break;
    case SPU_frest:  DST = spu_frest(A); break;
    case SPU_frsqest:DST = spu_frsqest(A); break;
    case SPU_fi:     DST = spu_fi(A,B); break;
    case SPU_fcmgt:  DST = spu_fcmgt(A,B); break;
    case SPU_fcmeq:  DST = spu_fcmeq(A,B); break;
    /* Double precision. SPU DP rounds to nearest and keeps denormals, while the
     * SPU run sets MXCSR to truncate + FTZ/DAZ for single precision -- switch for
     * the op. (Was unimplemented: a WWS job hit `dfs` at LS 0x816C and trapped.) */
#define SPU_DP(expr) do { unsigned _c = _mm_getcsr(); _mm_setcsr(_c & ~0xE040u); DST = (expr); _mm_setcsr(_c); } while (0)
    case SPU_dfa:   SPU_DP(spu_dfa(A,B)); break;
    case SPU_dfs:   SPU_DP(spu_dfs(A,B)); break;
    case SPU_dfm:   SPU_DP(spu_dfm(A,B)); break;
    case SPU_dfma:  SPU_DP(spu_dfma(A,B,DST)); break;
    case SPU_dfms:  SPU_DP(spu_dfms(A,B,DST)); break;
    case SPU_dfnms: SPU_DP(spu_dfnms(A,B,DST)); break;
    case SPU_dfnma: SPU_DP(spu_dfnma(A,B,DST)); break;
    case SPU_dfceq: DST = spu_dfceq(A,B); break;
    case SPU_dfcmeq:DST = spu_dfcmeq(A,B); break;
    case SPU_dfcgt: DST = spu_dfcgt(A,B); break;
    case SPU_dfcmgt:DST = spu_dfcmgt(A,B); break;
    case SPU_dftsv: DST = spu_dftsv(A,(int32_t)I); break;
    case SPU_bisled: { SPU_DRY_GUARD(); uint32_t t = PREF(A) & 0x3FFFC; DST = spu_link(pc + 4);
                       if ((ctx->event_status & ctx->event_mask) != 0) next = t; break; }
    case SPU_fesd:   DST = spu_fesd(A); break;
    case SPU_frds:   DST = spu_frds(A); break;
    case SPU_fma:  DSTC = spu_fma(A,B,T); break;    /* RRR: dest=rc */
    case SPU_fms:  DSTC = spu_fms(A,B,T); break;
    case SPU_fnms: DSTC = spu_fnms(A,B,T); break;
    /* immediate byte/halfword compares */
    case SPU_ceqbi: DST = spu_ceqbi(A,I); break;
    case SPU_cgtbi: DST = spu_cgtbi(A,I); break;
    case SPU_clgtbi:DST = spu_clgtbi(A,I); break;
    case SPU_ceqhi: DST = spu_ceqhi(A,I); break;
    case SPU_cgthi: DST = spu_cgthi(A,I); break;
    case SPU_clgthi:DST = spu_clgthi(A,I); break;
    /* integer / byte extras */
    case SPU_bg:    DST = spu_bg(A,B); break;
    case SPU_absdb: DST = spu_absdb(A,B); break;
    case SPU_avgb:  DST = spu_avgb(A,B); break;
    case SPU_sumb:  DST = spu_sumb(A,B); break;
    case SPU_mpys:  DST = spu_mpys(A,B); break;
    case SPU_mpyhh: DST = spu_mpyhh(A,B); break;
    case SPU_mpyhhu:DST = spu_mpyhhu(A,B); break;
    /* mpyhha / mpyhhau are RR-form (op11 0x346 / 0x34E): rt is BOTH the
     * accumulator and the destination. This used to be treated as RRR -- the
     * accumulator came from the low 7 bits (rt, correct by luck) but the result
     * went to "rc" = bits 21-27, which for an RR op are opcode bits (r70) -- so
     * rt never got the sum and r70 was clobbered. 2026-10-03. */
    case SPU_mpyhha: DST = spu_mpyhha(A,B,DST); break;
    case SPU_mpyhhau:DST = spu_mpyhhau(A,B,DST); break;
    case SPU_bgx:    DST = spu_bgx(A,B,DST); break;
    /* halfword shifts right that had no case (decoded, then halted as unimplemented) */
    case SPU_rothmi: DST = spu_rothmi(A,I); break;
    case SPU_rotmahi:DST = spu_rotmahi(A,I); break;
    case SPU_rotmah: { u128 r; for (int i = 0; i < 8; i++) { uint32_t sh = (0u - B._u16[i]) & 0x1F;
                         r._s16[i] = (sh > 15) ? (int16_t)(A._s16[i] >> 15) : (int16_t)(A._s16[i] >> sh); }
                       DST = r; break; }
    /* rotate / shift extras */
    case SPU_rothm:    DST = spu_rothm(A,B); break;
    case SPU_rotqmbi:  DST = spu_rotqmbi(A,B); break;
    case SPU_rotqmbii: DST = spu_rotqmbii(A,(int)I); break;
    case SPU_rotqbybi: DST = spu_rotqbybi(A,B); break;
    case SPU_rotqmbybi:DST = spu_rotqmbybi(A,B); break;
    case SPU_shlqbybi: DST = spu_shlqbybi(A,B); break;
    /* channels */
    case SPU_wrch: SPU_DRY_GUARD(); spu_wrch(ctx, d.ch, DST); break;
    case SPU_rdch: SPU_DRY_GUARD(); DST = spu_rdch(ctx, d.ch); break;
    case SPU_rchcnt: SPU_DRY_GUARD(); DST = spu_splat_u32(spu_rchcnt(ctx, d.ch)); break;
    /* hints / no-ops */
    case SPU_nop: case SPU_lnop: case SPU_sync: case SPU_dsync:
    case SPU_hbr: case SPU_hbra: case SPU_hbrr: case SPU_mtspr:
    case SPU_mfspr: case SPU_fscrrd: case SPU_fscrwr: break;
    /* control flow */
    case SPU_br: case SPU_bra: next = d.tgt; break;
    case SPU_brsl: case SPU_brasl: DST = spu_link(pc + 4); next = d.tgt;
        /* v98f: the call-trace ring is only ever printed with SPU_CALLTRACE (which keeps the
         * slow loop); the fast loop skips the shared-global stores. */
        if (!cached) { extern void spu_trace_call(uint32_t,uint32_t); spu_trace_call(pc, d.tgt); } break;
    case SPU_brz:  if (PREF(DST) == 0) next = d.tgt; break;
    case SPU_brnz: if (PREF(DST) != 0) next = d.tgt; break;
    case SPU_brhz: if ((PREF(DST) & 0xFFFF) == 0) next = d.tgt; break;
    case SPU_brhnz:if ((PREF(DST) & 0xFFFF) != 0) next = d.tgt; break;
    case SPU_bi:   next = PREF(A) & 0x3FFFC; break;
    /* Read the target BEFORE writing the link: `bisl $0,$0` (the WWS jobs'
     * JobApi calls) has rt == ra, and writing first made the call branch
     * to its own return address, i.e. silently skip the callee. */
    case SPU_bisl: next = PREF(A) & 0x3FFFC; DST = spu_link(pc + 4); break;
    case SPU_iret: SPU_DRY_GUARD(); next = ctx->srr0 & 0x3FFFC; break;
    case SPU_biz:  if (PREF(DST) == 0) next = PREF(A) & 0x3FFFC; break;
    case SPU_binz: if (PREF(DST) != 0) next = PREF(A) & 0x3FFFC; break;
    case SPU_bihz: if ((PREF(DST) & 0xFFFF) == 0) next = PREF(A) & 0x3FFFC; break;
    case SPU_bihnz:if ((PREF(DST) & 0xFFFF) != 0) next = PREF(A) & 0x3FFFC; break;
    case SPU_stop: case SPU_stopd: SPU_DRY_GUARD();
        ctx->pc = next; ctx->stop_code = (uint32_t)I;
        ctx->status = SPU_STATUS_STOPPED_BY_STOP; return 1;
    /* conditional halts (assertions): stop the SPU when the condition holds,
     * else continue. The preferred word of ra is compared. */
    case SPU_heq: case SPU_heqi: case SPU_hgt: case SPU_hgti:
    case SPU_hlgt: case SPU_hlgti: { SPU_DRY_GUARD();
        uint32_t a = PREF(A), rhs = (d.op==SPU_heq||d.op==SPU_hgt||d.op==SPU_hlgt) ? PREF(B) : (uint32_t)I;
        int hit = (d.op==SPU_heq||d.op==SPU_heqi)   ? (a == rhs)
                : (d.op==SPU_hgt||d.op==SPU_hgti)   ? ((int32_t)a > (int32_t)rhs)
                :                                     (a > rhs);      /* hlgt/hlgti */
        if (hit) { extern void spu_trace_dump(uint32_t); spu_trace_dump(pc);
                   ctx->pc = pc; ctx->stop_code = 0x1000; ctx->status = SPU_STATUS_STOPPED_BY_HALT; return 1; }
        break; }
    default: SPU_DRY_GUARD();
        fprintf(stderr, "[spu_interp] unimplemented op '%s' (0x%08X) at LS 0x%05X\n",
                d.op < SPU_OP_COUNT ? spu_op_name[d.op] : "?", insn, pc);
        ctx->pc = pc; ctx->status = SPU_STATUS_STOPPED_BY_HALT; return 1;
    }
    ctx->pc = next;
    return 0;
}
static __attribute__((noinline)) int spu_step(spu_context* ctx) {
    uint32_t pc = ctx->pc, m = pc & SPU_LS_MASK;
    const uint8_t* p = ctx->ls;
    uint32_t insn = ((uint32_t)p[m] << 24) | ((uint32_t)p[m+1] << 16)
                  | ((uint32_t)p[m+2] << 8) | p[m+3];
    spu_ins d; spu_decode1(insn, pc, &d);
    return spu_step_impl(ctx, pc, insn, d, 0);
}

uint32_t g_spu_interp_last_pc = 0;
uint64_t g_spu_interp_steps   = 0;

/* Call-trace ring buffer for diagnosing SPU asserts (env SPU_CALLTRACE). */
#define SPU_TRACE_N 32
static struct { uint32_t from, to; } s_trace[SPU_TRACE_N];
static unsigned s_trace_i = 0;
void spu_trace_call(uint32_t from, uint32_t to) {
    s_trace[s_trace_i % SPU_TRACE_N].from = from;
    s_trace[s_trace_i % SPU_TRACE_N].to   = to;
    s_trace_i++;
}
void spu_trace_dump(uint32_t at) {
    if (!getenv("SPU_CALLTRACE")) return;
    fprintf(stderr, "[SPU-TRACE] halt at LS 0x%05X; last %d calls (from->to):\n", at, SPU_TRACE_N);
    unsigned n = s_trace_i < SPU_TRACE_N ? s_trace_i : SPU_TRACE_N;
    for (unsigned k = 0; k < n; k++) {
        unsigned idx = (s_trace_i - n + k) % SPU_TRACE_N;
        fprintf(stderr, "   0x%05X -> 0x%05X\n", s_trace[idx].from, s_trace[idx].to);
    }
}

uint32_t spu_interp_run(spu_context* ctx, uint32_t start_lsa) {
    return spu_interp_run_until(ctx, start_lsa, 0);
}

static uint32_t spu_interp_run_until_impl(spu_context* ctx, uint32_t start_lsa, uint32_t stop_lsa);
static uint32_t spu_interp_run_until_fast(spu_context* ctx, uint32_t start_lsa, uint32_t stop_lsa);
/* v98f fast interpreter loop. The SPU interpreter runs code that has no lift: in inFamous mainly
 * the per-batch code fragments that BatchJobList's job module (overlay 209) calls through
 * pointers in its data buffers (bisl at 0xA620/0xA674/0xA6C0) -- they carry per-instance constants
 * and sit at varying LS addresses, so they cannot be lifted ahead of time. w2 spent 50-60% of its
 * lane time there at ~100 ns per SPU instruction. The fast loop executes the SAME spu_step code
 * (same helpers, same semantics) but (1) caches decodes per (pc, word), (2) tests a bitmap of
 * registered lifted entries before the per-step rejoin lookup (registry walk on every miss),
 * (3) does no per-step stores to shared globals (g_spu_interp_last_pc / the call-trace ring were
 * written on every step/call by all interpreting lanes at once -> cache-line ping-pong), and
 * (4) skips the per-step debug-env checks (any debug env set -> old loop).
 * SPU_INTERP_FAST=0 -> old loop; PS3_AB phase 1 (g_ab_mask & 1) -> old loop.
 * SPU_INTERP_DIFF=N: the first N interpreter runs execute both loops dry from the same state and
 * compare all GPRs + the whole LS + pc/status ([INTERP-DIFF] summary lines). */
extern volatile int g_ab_mask;
static int s_fast_ok = -1;
static int spu_interp_fast_ok(void) {
    if (__builtin_expect(s_fast_ok < 0, 0)) {
        const char* e = getenv("SPU_INTERP_FAST");
        int ok = !(e && *e == '0');
        static const char* const dbg[] = { "YDKJ_SPU_TRACE", "SPU_STEPCAP", "YDKJ_CRI_GATE1TRACE",
            "SPU_INTERP_PCLOG", "SPU_INTERP_REGDUMP_PC", "WWS_JOBAPI", "SPU_INTERP_DELTA", "SPU_CALLTRACE" };
        for (unsigned i = 0; i < sizeof dbg / sizeof dbg[0]; i++) if (getenv(dbg[i])) ok = 0;
        fprintf(stderr, "[SPU] v98f fast interpreter loop: %s (SPU_INTERP_FAST=0 off; PS3_AB phase 1 off)\n", ok ? "ON" : "OFF");
        s_fast_ok = ok;
    }
    return s_fast_ok;
}
static void spu_interp_diff(spu_context* ctx, uint32_t start_lsa, uint32_t stop_lsa);
static volatile long s_diff_left = -1;
/* v96a [LANEPROF]: interpreter time inside policy-module runs. */
uint32_t spu_interp_run_until(spu_context* ctx, uint32_t start_lsa, uint32_t stop_lsa) {
    uint64_t t0 = spu_lp_begin();
    uint32_t r;
    int fast = spu_interp_fast_ok() && !(g_ab_mask & 1);
    if (__builtin_expect(s_diff_left != 0, 0)) {
        if (s_diff_left < 0) { const char* e = getenv("SPU_INTERP_DIFF"); s_diff_left = e ? atol(e) : 0; }
        static long s_minpc = -1; if (s_minpc < 0) { const char* e = getenv("SPU_INTERP_DIFF_MINPC"); s_minpc = e ? strtol(e, 0, 16) : 0; }
        if (s_diff_left > 0 && spu_interp_fast_ok() && (start_lsa & 0x3FFFF) >= (uint32_t)s_minpc) spu_interp_diff(ctx, start_lsa, stop_lsa);
    }
    r = fast ? spu_interp_run_until_fast(ctx, start_lsa, stop_lsa)
             : spu_interp_run_until_impl(ctx, start_lsa, stop_lsa);
    spu_lp_end(t0, LP_INTERP);
    return r;
}

extern uint8_t g_spu_fn_bitmap[0x40000 / 32];   /* spu_channels.c: bit set = some image registered a fn at that LS word */
extern volatile uint32_t g_spu_fn_reg_gen;       /* spu_channels.c: bumped on every registration (v98f stage 2) */
#define SPU_DCF_LIFTED  1u   /* the bitmap bit for this pc was set (or pc >= 256 KB): do the real lookup */
#define SPU_DCF_A70     2u   /* pc == 0xA70 (taskset syscall HLE check) */
#define SPU_DCF_PLUG    4u   /* v135: a compiled WWS plugin function starts here (spu_plug_lookup, ABX 0x10) */
/* v135: the interpreter rejoins compiled WWS plugin code (infamous-recomp/spu/wwsplug) at any plugin function start.
 * Without this, once the interpreter owned a plugin call -- e.g. it was running a job-module lift gap after an
 * interrupt unwind and that code called the plugin -- it interpreted the whole plugin (play134/135: interpreter time
 * unchanged with the plugins compiled). Checked only where the decode cache is filled (once per pc+word, pc >= 64 KB)
 * and only while ABX 0x10 is on; never on a run's first instruction (guaranteed progress). */
extern volatile int g_abx;
typedef void (*spu_plug_fn_t)(spu_context*);
extern spu_plug_fn_t spu_plug_lookup(spu_context* ctx, uint32_t pc);
volatile long g_spu_plug_rejoins;   /* v135: interpreter -> compiled plugin hand-backs ([plug] line) */
static uint32_t spu_interp_run_until_fast(spu_context* ctx, uint32_t start_lsa, uint32_t stop_lsa) {
    ctx->pc = start_lsa & 0x3FFFC;
    ctx->status = SPU_STATUS_RUNNING;
    uint64_t steps = 0;
    for (;;) {
        uint32_t pc = ctx->pc, m = pc & SPU_LS_MASK;
        uint32_t insn; memcpy(&insn, ctx->ls + m, 4);
#if defined(__clang__) || defined(__GNUC__)
        insn = __builtin_bswap32(insn);
#else
        insn = _byteswap_ulong(insn);
#endif
        spu_dc_ent* ce = &t_spu_dc[(m >> 2) & (SPU_DC_N - 1)];
        const uint32_t gen = g_spu_fn_reg_gen;
        if (__builtin_expect(!(ce->pc == (pc | 1u) && ce->insn == insn && ce->gen == gen), 0)) {
            spu_decode1(insn, pc, &ce->d);
            ce->pc = pc | 1u; ce->insn = insn; ce->gen = gen;
            const uint32_t lif = (pc >= 0x40000u || (g_spu_fn_bitmap[pc >> 5] & (1u << ((pc >> 2) & 7)))) ? SPU_DCF_LIFTED : 0u;
            /* v137: any LS address (fixed job-manager stubs live at 0xA2C / 0x3FEC0), only while ABX 0x10 is on; the
             * lookup's first-word bitmap makes the common miss a single bit test */
            ce->flags = lif | (pc == 0xA70u ? SPU_DCF_A70 : 0u)
                      | ((!lif && (g_abx & 0x10) && spu_plug_lookup(ctx, pc)) ? SPU_DCF_PLUG : 0u);
        }
        const uint32_t fl = ce->flags;
        if (__builtin_expect(fl & SPU_DCF_A70, 0)) {   /* identical to the old loop */
            if (t_spu_dry) { t_spu_dry_abort = 1; ctx->status = SPU_STATUS_STOPPED_BY_HALT; return 0; }
            uint32_t sc = ((uint32_t)ctx->ls[0x27C4]<<24)|((uint32_t)ctx->ls[0x27C5]<<16)
                        | ((uint32_t)ctx->ls[0x27C6]<<8) | ctx->ls[0x27C7];
            if (ctx->image_id == 22 || (ctx->policy_mode && sc == 0xA70u)) {
                extern void spu_spurs_taskset_syscall(spu_context*);
                int _save = ctx->image_id; ctx->image_id = 22;
                spu_spurs_taskset_syscall(ctx);
                ctx->image_id = _save;
                ctx->pc = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
                continue;
            }
        }
        /* rejoin: same condition as the old loop; the bitmap only skips lookups that must miss
         * (registered addresses are exact keys; the flag is re-derived whenever the registry grew). */
        if (((stop_lsa && pc == stop_lsa) || ((fl & SPU_DCF_LIFTED) && spu_lifted_lookup(ctx, pc))) &&
            ctx->image_id >= 0) {
            g_spu_interp_steps = steps; g_spu_interp_last_pc = pc; return pc;
        }
        if ((fl & SPU_DCF_PLUG) && steps && (g_abx & 0x10) && ctx->image_id >= 0) {   /* v135 */
            g_spu_plug_rejoins++;
            g_spu_interp_steps = steps; g_spu_interp_last_pc = pc; return pc;
        }
        steps++;
        if (spu_step_impl(ctx, pc, insn, ce->d, 1)) {
            if (ctx->stop_code == 0x110 && s_spu_pending_evt_valid && !t_spu_dry_abort) {
                spu_channel_read(&ctx->ch_out_mbox);
                ctx->rcv_evt[0] = 0;
                ctx->rcv_evt[1] = s_spu_pending_evt[0];
                ctx->rcv_evt[2] = s_spu_pending_evt[1];
                ctx->rcv_evt[3] = s_spu_pending_evt[2];
                ctx->rcv_evt_n = 4; ctx->rcv_evt_i = 0;
                s_spu_pending_evt_valid = 0;
                ctx->status = SPU_STATUS_RUNNING;
                continue;
            }
            g_spu_interp_steps = steps; g_spu_interp_last_pc = pc;
            return ctx->stop_code;
        }
    }
}

static uint64_t spu_diff_now(void) {
#if defined(_MSC_VER) && !defined(__clang__)
    return __rdtsc();
#else
    return __builtin_ia32_rdtsc();
#endif
}
static void spu_interp_diff(spu_context* ctx, uint32_t start_lsa, uint32_t stop_lsa) {
    static volatile long lk;
    static long n_run, n_same, n_diff, n_abort; static uint64_t c_fast, c_slow, st_fast;
#if defined(_MSC_VER) && !defined(__clang__)
    if (_InterlockedDecrement(&s_diff_left) < 0) { s_diff_left = 0; return; }
#else
    if (__sync_sub_and_fetch(&s_diff_left, 1) < 0) { s_diff_left = 0; return; }
#endif
    static SPU_INTERP_TLS uint8_t* ls0; static SPU_INTERP_TLS uint8_t* lsA;
    if (!ls0) { ls0 = (uint8_t*)malloc(SPU_LS_SIZE); lsA = (uint8_t*)malloc(SPU_LS_SIZE); }
    if (!ls0 || !lsA) return;
    u128 g0[128], gA[128];
    memcpy(g0, ctx->gpr, sizeof g0); memcpy(ls0, ctx->ls, SPU_LS_SIZE);
    uint32_t pc0 = ctx->pc, st0 = ctx->status, sc0 = ctx->stop_code;
    uint32_t lp0 = g_spu_interp_last_pc; uint64_t gs0 = g_spu_interp_steps;
    t_spu_dry = 1; t_spu_dry_abort = 0;
    uint64_t a0 = spu_diff_now();
    uint32_t rA = spu_interp_run_until_fast(ctx, start_lsa, stop_lsa);
    uint64_t a1 = spu_diff_now();
    uint64_t stepsA = g_spu_interp_steps;
    int abA = t_spu_dry_abort; uint32_t pcA = ctx->pc, stA = ctx->status, scA = ctx->stop_code;
    memcpy(gA, ctx->gpr, sizeof gA); memcpy(lsA, ctx->ls, SPU_LS_SIZE);
    memcpy(ctx->gpr, g0, sizeof g0); memcpy(ctx->ls, ls0, SPU_LS_SIZE);
    ctx->pc = pc0; ctx->status = st0; ctx->stop_code = sc0;
    t_spu_dry_abort = 0;
    uint64_t b0 = spu_diff_now();
    uint32_t rB = spu_interp_run_until_impl(ctx, start_lsa, stop_lsa);
    uint64_t b1 = spu_diff_now();
    int abB = t_spu_dry_abort;
    int same = (rA == rB) && (abA == abB) && pcA == ctx->pc && stA == ctx->status && (abA || scA == ctx->stop_code) &&
               !memcmp(gA, ctx->gpr, sizeof gA) && !memcmp(lsA, ctx->ls, SPU_LS_SIZE);
    int bad_reg = -1; long bad_ls = -1;
    if (!same) { for (int g = 0; g < 128; g++) if (memcmp(&gA[g], &ctx->gpr[g], 16)) { bad_reg = g; break; }
                 for (long o = 0; o < SPU_LS_SIZE; o++) if (lsA[o] != ctx->ls[o]) { bad_ls = o; break; } }
    memcpy(ctx->gpr, g0, sizeof g0); memcpy(ctx->ls, ls0, SPU_LS_SIZE);
    ctx->pc = pc0; ctx->status = st0; ctx->stop_code = sc0;
    g_spu_interp_last_pc = lp0; g_spu_interp_steps = gs0;
    t_spu_dry = 0; t_spu_dry_abort = 0;
    while (__sync_lock_test_and_set(&lk, 1)) { }
    n_run++; if (abA || abB) n_abort++; if (same) n_same++; else n_diff++;
    if (!abA && !abB) { c_fast += a1 - a0; c_slow += b1 - b0; st_fast += stepsA; }
    long nr = n_run;
    if (!same && n_diff <= 16)
        fprintf(stderr, "[INTERP-DIFF] MISMATCH start=0x%05X stop=0x%05X img=%d ovl=%d rA=0x%X rB=0x%X ab=%d/%d pc=%05X/%05X st=%X/%X reg=%d ls=0x%05lX\n",
                start_lsa, stop_lsa, ctx->image_id, ctx->resident_ovl, rA, rB, abA, abB, pcA, ctx->pc, stA, ctx->status, bad_reg, bad_ls);
    if ((nr & (nr - 1)) == 0 || s_diff_left == 0)
        fprintf(stderr, "[INTERP-DIFF] runs %ld: identical %ld, mismatched %ld (dry-aborted %ld) | fast %.1f vs old %.1f cycles/step (%llu steps)\n",
                nr, n_same, n_diff, n_abort, st_fast ? (double)c_fast / st_fast : 0.0, st_fast ? (double)c_slow / st_fast : 0.0,
                (unsigned long long)st_fast);
    __sync_lock_release(&lk);
}
static uint32_t spu_interp_run_until_impl(spu_context* ctx, uint32_t start_lsa, uint32_t stop_lsa) {
    ctx->pc = start_lsa & 0x3FFFC;
    ctx->status = SPU_STATUS_RUNNING;
    uint64_t steps = 0;
    /* YDKJ_SPU_TRACE=N: log the last N PCs into a ring buffer and dump them when the
     * interp halts -- shows the path to a branch-to-0 (the cri task/policy wall). */
    static int _tr=-1; if(_tr<0){const char*e=getenv("YDKJ_SPU_TRACE");_tr=e?atoi(e):0;}
    static uint64_t _cap=0; { static int _ci=0; if(!_ci){_ci=1; const char*e=getenv("SPU_STEPCAP"); _cap=e?strtoull(e,0,0):0;} }
    /* YDKJ_CRI_GATE1TRACE=N: cri decode task busy-spins in the validator func_00026E80
     * (LS 0x26E80..0x26F14), a straight-line leaf that returns r3 = 0 / 0x8041090F /
     * 0x80410909. Hand-decoding its selb/fsm/gb/ceqh lanes proved unreliable, so log the
     * ACTUAL runtime result + inputs the first N times it returns. Fires on range-exit
     * (pc left the fn), when r3 and the leaf's non-restored intermediates are still live. */
    static int _g1=-1; if(_g1<0){const char*e=getenv("YDKJ_CRI_GATE1TRACE");_g1=e?atoi(e):0;}
    static int _g1in=0;
    uint32_t ring[64]; int rc=0, rn=0;
    for (;;) {
        /* Match the lifted path: only synthetic HLE tasks may bypass the
         * resident policy. Real tasksets also store 0xA70 in their API table.
         * Preserve the legacy CRI return convention while invoking the HLE. */
        if (ctx->pc == 0xA70u) {
            uint32_t sc = ((uint32_t)ctx->ls[0x27C4]<<24)|((uint32_t)ctx->ls[0x27C5]<<16)
                        | ((uint32_t)ctx->ls[0x27C6]<<8) | ctx->ls[0x27C7];
            if (ctx->image_id == 22 || (ctx->policy_mode && sc == 0xA70u)) {
                extern void spu_spurs_taskset_syscall(spu_context*);
                int _save = ctx->image_id; ctx->image_id = 22;
                spu_spurs_taskset_syscall(ctx);
                ctx->image_id = _save;
                ctx->pc = ctx->gpr[0]._u32[0] & SPU_LS_MASK;   /* resume at link r0 */
                continue;
            }
        }
        /* Rejoin the compiled fast path only for images that HAVE lifted
         * functions (image_id >= 0). image_id < 0 = pure interpretation: an
         * un-lifted image (e.g. a title's raw SPU jobs) must never rejoin
         * another image's functions that happen to share an LS address. */
        /* ponytail: pc-only match; a recursive call through the same site would stop early -- compare r1 too if one shows up. */
        if (ctx->image_id >= 0 && ((stop_lsa && ctx->pc == stop_lsa) || spu_lifted_lookup(ctx, ctx->pc))) { g_spu_interp_steps = steps; g_spu_interp_last_pc = ctx->pc; return ctx->pc; }  /* rejoin fast path */
        g_spu_interp_last_pc = ctx->pc;
        /* SPU_INTERP_PCLOG=<file>: every interpreted pc + r3 preferred word
         * (offline job replay / divergence hunting; never set in normal runs). */
        { static FILE* s_pcl = (FILE*)1; if (s_pcl == (FILE*)1) { const char* e = getenv("SPU_INTERP_PCLOG"); s_pcl = e ? fopen(e, "w") : 0; }
          if (s_pcl) fprintf(s_pcl, "%05X r3=%08X r1=%05X\n", ctx->pc & 0x3FFFF, ctx->gpr[3]._u32[0], ctx->gpr[1]._u32[0] & 0x3FFFF); }
        /* SPU_INTERP_REGDUMP_PC=<hex pc>[:<n>]: dump r0-r127 + LS 0x26000-0x26200 the
         * first n (default 2) times the interpreter reaches that pc. */
        { static int s_rdn = -2; static uint32_t s_rdpc = 0;
          if (s_rdn == -2) { const char* e = getenv("SPU_INTERP_REGDUMP_PC"); s_rdn = 0;
              if (e) { char* q; s_rdpc = (uint32_t)strtoul(e, &q, 16); s_rdn = (*q == ':') ? atoi(q + 1) : 2; } }
          static uint32_t s_rdf = 1; if (s_rdf == 1) { const char* e = getenv("WWS_JOBDUMP_SPURS"); s_rdf = e ? (uint32_t)strtoul(e, 0, 16) : 0; }
          if (s_rdn > 0 && (ctx->pc & 0x3FFFF) == s_rdpc &&
              (!s_rdf || (((uint32_t)ctx->ls[0x133C] << 24) | ((uint32_t)ctx->ls[0x133D] << 16) |
                          ((uint32_t)ctx->ls[0x133E] << 8) | ctx->ls[0x133F]) == s_rdf)) { s_rdn--;
              fprintf(stderr, "[regdump] pc=0x%05X img=%d\n", s_rdpc, ctx->image_id);
              for (int g = 0; g < 128; g++)
                  fprintf(stderr, "[regdump]  r%-3d=%08X %08X %08X %08X\n", g, ctx->gpr[g]._u32[0], ctx->gpr[g]._u32[1], ctx->gpr[g]._u32[2], ctx->gpr[g]._u32[3]);
              for (uint32_t o = 0x26000; o < 0x26200; o += 16)
                  fprintf(stderr, "[regdump]  LS %05X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X %02X%02X%02X%02X\n", o,
                      ctx->ls[o],ctx->ls[o+1],ctx->ls[o+2],ctx->ls[o+3],ctx->ls[o+4],ctx->ls[o+5],ctx->ls[o+6],ctx->ls[o+7],
                      ctx->ls[o+8],ctx->ls[o+9],ctx->ls[o+10],ctx->ls[o+11],ctx->ls[o+12],ctx->ls[o+13],ctx->ls[o+14],ctx->ls[o+15]);
              fflush(stderr); } }
        { static int s_jon = -1; if (s_jon < 0) s_jon = getenv("WWS_JOBAPI") ? 1 : 0;
          if (s_jon && ctx->policy_mode) { extern void spu_wws_jobapi_trace(spu_context*); spu_wws_jobapi_trace(ctx); } }
        if (_g1>0) {
            int inr = (ctx->pc >= 0x26E80u && ctx->pc < 0x26F14u);
            if (inr) { _g1in = 1; }
            else if (_g1in) {
                _g1in = 0;
                #define GLB(o) (((uint32_t)ctx->ls[(o)]<<24)|((uint32_t)ctx->ls[(o)+1]<<16)|((uint32_t)ctx->ls[(o)+2]<<8)|ctx->ls[(o)+3])
                fprintf(stderr,"[gate1] func_00026E80 ret r3=%08X | LS2FB0=%08X LS2FB4=%08X LS2FB8=%08X | LS1E0=%08X LS1E4=%08X LS1E8=%08X | r1=%05X r29=%08X r32=%08X r16=%08X r22=%08X r8=%08X r6=%08X\n",
                    ctx->gpr[3]._u32[0], GLB(0x2FB0),GLB(0x2FB4),GLB(0x2FB8), GLB(0x1E0),GLB(0x1E4),GLB(0x1E8),
                    ctx->gpr[1]._u32[0], ctx->gpr[29]._u32[0], ctx->gpr[32]._u32[0],
                    ctx->gpr[16]._u32[0], ctx->gpr[22]._u32[0], ctx->gpr[8]._u32[0], ctx->gpr[6]._u32[0]);
                #undef GLB
                fflush(stderr); _g1--;
            }
        }
        if (_tr>0) { ring[rc&63]=ctx->pc; rc++; if(rn<64)rn++; }
        steps++;
        /* SPU_STEPCAP=N: a task that never halts (infinite work/wait loop) never
         * dumps its ring. Force a one-shot dump after N steps to see where it loops. */
        if (_tr>0 && _cap && steps == _cap) {
            fprintf(stderr,"[spu-trace] STEPCAP pc=0x%05X after %llu steps; last %d PCs:",
                    ctx->pc, (unsigned long long)steps, rn);
            for(int k=rn;k>0;k--) fprintf(stderr," %05X", ring[(rc-k)&63]);
            fprintf(stderr,"\n");
            /* dump the LS 0x2700 SpursTasksetContext the poll reads -- compare to the
             * working RPCS3 SPU0 dump (RUNNING/READY/ENABLED=0x80000000, SIGNALLED=0). */
            #define LB(o) (((uint32_t)ctx->ls[(o)]<<24)|((uint32_t)ctx->ls[(o)+1]<<16)|((uint32_t)ctx->ls[(o)+2]<<8)|ctx->ls[(o)+3])
            fprintf(stderr,"[spu-trace] LS ctx: RUN=%08X READY=%08X PEND=%08X ENA=%08X SIG=%08X WAIT=%08X | 2790=%08X 2794=%08X 27D8=%08X 2FB0=%08X\n",
                LB(0x2700),LB(0x2710),LB(0x2720),LB(0x2730),LB(0x2740),LB(0x2750),LB(0x2790),LB(0x2794),LB(0x27D8),LB(0x2FB0));
            #undef LB
            fflush(stderr); _tr--; g_spu_interp_steps=steps; return 0x2000u;
        }
        /* SPU_INTERP_DELTA=<file>[:<maxlines>]: per-step register delta trace
         * (pc, insn word, every GPR that changed) for contexts passing the
         * WWS_JOBDUMP_SPURS filter -- an exact execution trace of one job. */
        static FILE* s_dt = (FILE*)1; static long s_dt_left = 0; static uint32_t s_dtf = 1;
        if (s_dt == (FILE*)1) { const char* e = getenv("SPU_INTERP_DELTA"); s_dt = 0;
            if (e) { char fn[260]; const char* c = strchr(e, ':'); size_t n = c ? (size_t)(c - e) : strlen(e);
                if (n >= sizeof fn) n = sizeof fn - 1; memcpy(fn, e, n); fn[n] = 0;
                s_dt_left = c ? atol(c + 1) : 200000; s_dt = fopen(fn, "w"); }
            const char* f = getenv("WWS_JOBDUMP_SPURS"); s_dtf = f ? (uint32_t)strtoul(f, 0, 16) : 0; }
        static int s_dtw = -2; if (s_dtw == -2) { const char* e = getenv("WWS_JOBDUMP_WID"); s_dtw = e ? atoi(e) : -1; }
        int dt_this = s_dt && s_dt_left > 0 && ctx->policy_mode &&
            (s_dtw < 0 || (((int)ctx->ls[0x1340] << 8) | ctx->ls[0x1341]) == s_dtw) &&
            (!s_dtf || (((uint32_t)ctx->ls[0x133C] << 24) | ((uint32_t)ctx->ls[0x133D] << 16) |
                        ((uint32_t)ctx->ls[0x133E] << 8) | ctx->ls[0x133F]) == s_dtf);
        static u128 s_dt_before[128]; uint32_t dt_pc = ctx->pc & 0x3FFFF, dt_w = 0;
        if (dt_this) { memcpy(s_dt_before, ctx->gpr, sizeof s_dt_before);
            dt_w = ((uint32_t)ctx->ls[dt_pc] << 24) | ((uint32_t)ctx->ls[dt_pc+1] << 16) | ((uint32_t)ctx->ls[dt_pc+2] << 8) | ctx->ls[dt_pc+3]; }
        int dt_stop = spu_step(ctx);
        if (dt_this) { s_dt_left--;
            fprintf(s_dt, "%05X %08X", dt_pc, dt_w);
            for (int g = 0; g < 128; g++) if (memcmp(&s_dt_before[g], &ctx->gpr[g], 16))
                fprintf(s_dt, " r%d=%08X.%08X.%08X.%08X", g, ctx->gpr[g]._u32[0], ctx->gpr[g]._u32[1], ctx->gpr[g]._u32[2], ctx->gpr[g]._u32[3]);
            fputc('\n', s_dt); if ((s_dt_left & 255) == 0) fflush(s_dt); }
        if (dt_stop) {
            /* stop 0x110 = SYS_SPU_THREAD_STOP_RECEIVE_EVENT. The worker writes
             * the SPU queue number to its out-mailbox, stops, and lv2 replies
             * with {CELL_OK, data1, data2, data3} in the in-mailbox -- which is
             * how these sim SPUs receive each frame's work-descriptor EA.
             * Treating the stop as "job finished" let them resume with nothing
             * in that structure and DMA their results to address 0.
             * Service it and keep running; with no event pending, fall through
             * to the old behaviour so nothing waits forever. */
            if (ctx->stop_code == 0x110 && s_spu_pending_evt_valid) {
                spu_channel_read(&ctx->ch_out_mbox);      /* the SPU queue key */
                ctx->rcv_evt[0] = 0;                      /* CELL_OK */
                ctx->rcv_evt[1] = s_spu_pending_evt[0];
                ctx->rcv_evt[2] = s_spu_pending_evt[1];
                ctx->rcv_evt[3] = s_spu_pending_evt[2];
                ctx->rcv_evt_n = 4; ctx->rcv_evt_i = 0;
                s_spu_pending_evt_valid = 0;
                ctx->status = SPU_STATUS_RUNNING;
                continue;
            }
            g_spu_interp_steps = steps;
            if (_tr>0) { fprintf(stderr,"[spu-trace] halt stop=0x%X pc=0x%05X after %llu steps; last %d PCs:",
                    ctx->stop_code, ctx->pc, (unsigned long long)steps, rn);
                for(int k=rn;k>0;k--) fprintf(stderr," %05X", ring[(rc-k)&63]);
                fprintf(stderr,"\n"); fflush(stderr); _tr--; }
            return ctx->stop_code; }
    }
}

void spu_dispatch(spu_context* ctx, uint32_t target) {
    for (;;) {
        target &= 0x3FFFC;
        spu_lifted_fn fn = spu_lifted_lookup(ctx, target);
        if (fn) { ctx->pc = target; fn(ctx); return; }
        uint32_t next = spu_interp_run(ctx, target);
        if (ctx->status & (SPU_STATUS_STOPPED_BY_STOP | SPU_STATUS_STOPPED_BY_HALT))
            return;
        if ((next & 0x3FFFC) == target) return;   /* no-progress guard */
        target = next;
    }
}
