/*
 * ps3recomp - RSX Command Buffer Processor
 *
 * Parses NV47xx GPU methods from the command buffer and updates RSX state.
 * Dispatches to the registered graphics backend for actual rendering.
 *
 * Command buffer format (NV47xx FIFO):
 *   Each command is a 32-bit header followed by N data words.
 *   Header format: [31:29] type | [28:18] count | [17:13] subchannel | [12:2] method | [1:0] flags
 *
 *   Type 0 (increasing): method, method+4, method+8, ... for each data word
 *   Type 2 (non-increasing): same method repeated for each data word
 *   Type 1 (jump): jump to address in data
 *   Type 3 (call/return): call/return from subroutine
 */

#include "rsx_commands.h"
#include "ps3emu/milestone.h"   /* ps3_ms -- boot milestone log */
#include <stdio.h>
#include <stdlib.h>   /* getenv -- an implicit decl returns int, truncating the pointer */
#include "rsx_getenv_cache.h"   /* cached getenv() for hot paths (2026-10-04) */
#include <string.h>
/* v33: set by the D3D12 backend when async vertex conversion is on; called before any
 * RSX-side write the guest can observe (labels, semaphores, reference, 2D transfers). */
void (*g_rsx_vtx_sync)(void) = 0;
#include "../../runtime/memory/vm.h"    /* VM_HLE_INJECT_BASE */

/* ---------------------------------------------------------------------------
 * Global backend
 * -----------------------------------------------------------------------*/

static rsx_backend* s_backend = NULL;

/* Last NV406E_SET_REFERENCE value seen in the FIFO. The title writes a
 * reference then spins on the GCM control register's `ref` field until the RSX
 * reports it as reached; cellGcmSys mirrors this into the guest control
 * register after each drain so those waits (cellGcmFinish / wait-label) unblock. */
u32 g_rsx_last_reference = 0;

void rsx_set_backend(rsx_backend* backend)
{
    s_backend = backend;
}

rsx_backend* rsx_get_backend(void)
{
    return s_backend;
}

/* ---------------------------------------------------------------------------
 * State initialization
 * -----------------------------------------------------------------------*/

void rsx_state_init(rsx_state* state)
{
    memset(state, 0, sizeof(rsx_state));
    /* RSX reset value for the constant vertex attributes is (0,0,0,1); a zeroed
     * struct would leave w at 0. */
    for (int _i = 0; _i < RSX_MAX_VERTEX_ATTRIBS; _i++)
        state->vertex_data4f[_i][3] = 1.0f;

    /* Default viewport */
    state->viewport_w = 1280;
    state->viewport_h = 720;
    state->clip_min = 0.0f;
    state->clip_max = 1.0f;
    state->depth_bounds_min = 0.0f;   /* v35 */
    state->depth_bounds_max = 1.0f;

    /* Default scissor */
    state->scissor_w = 4096;
    state->scissor_h = 4096;

    /* Default depth */
    state->depth_func = 1; /* LESS */
    state->depth_mask = 1;

    /* Default cull */
    state->cull_face = 1; /* BACK */
    state->front_face = 0; /* CW */

    /* Default color mask: all channels writable (A|R|G|B) */
    state->color_mask = 0x01010101;

    /* Default stencil: sensible initial values */
    state->stencil_func = 0x0207; /* ALWAYS */
    state->stencil_ref = 0;
    state->stencil_mask = 0xFF;
    state->stencil_op_fail = 0x1E00;  /* KEEP */
    state->stencil_op_zfail = 0x1E00; /* KEEP */
    state->stencil_op_zpass = 0x1E00; /* KEEP */
    state->stencil_wmask = 0xFF; state->bstencil_wmask = 0xFF; state->bstencil_func = 0x0207;
    state->bstencil_mask = 0xFF; state->bstencil_op_fail = state->bstencil_op_zfail = state->bstencil_op_zpass = 0x1E00;

    /* Default alpha test */
    state->alpha_func = 0x0207; /* ALWAYS */
    state->alpha_ref = 0;

    /* Default shader control: 32-bit colour exports (r0..) -- matches every
     * title that never programs the register. */
    state->shader_control = CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS;

    /* Mark everything dirty */
    state->surface_dirty = 1;
    state->viewport_dirty = 1;
    state->blend_dirty = 1;
    state->depth_dirty = 1;
    state->stencil_dirty = 1;
    state->texture_dirty = 1;
    state->vertex_dirty = 1;
    state->color_mask_dirty = 1;
    state->alpha_dirty = 1;
    state->shader_dirty = 1;
}

/* ---------------------------------------------------------------------------
 * Method processing
 * -----------------------------------------------------------------------*/

static int process_surface_method(rsx_state* state, u32 method, u32 data)
{
    switch (method) {
    case NV4097_SET_SURFACE_FORMAT:
        state->surface_format = data;
        state->surface_dirty = 1;
        return 0;
    case NV4097_SET_SURFACE_CLIP_HORIZONTAL:
        state->surface_clip_x = data & 0xFFFF;
        state->surface_clip_w = (data >> 16) & 0xFFFF;
        state->surface_dirty = 1;
        return 0;
    case NV4097_SET_SURFACE_CLIP_VERTICAL:
        state->surface_clip_y = data & 0xFFFF;
        state->surface_clip_h = (data >> 16) & 0xFFFF;
        state->surface_dirty = 1;
        return 0;
    case NV4097_SET_SURFACE_COLOR_AOFFSET:
        state->surface_color_offset[0] = data;
        { static int _s = -1; if (_s < 0) _s = getenv("SURFDBG") ? 1 : 0;
          static unsigned _seen[32]; static int _n = 0;
          if (_s) { int f = 0; for (int k = 0; k < _n; k++) if (_seen[k] == data) f = 1;
              if (!f && _n < 32) { _seen[_n++] = data;
                  fprintf(stderr, "[SURF] color offset -> 0x%08X%c", data, 10); } } }
        state->surface_dirty = 1;
        return 0;
    case NV4097_SET_SURFACE_COLOR_BOFFSET:
        state->surface_color_offset[1] = data;
        state->surface_dirty = 1;
        return 0;
    case NV4097_SET_SURFACE_COLOR_COFFSET:
        state->surface_color_offset[2] = data;
        state->surface_dirty = 1;
        return 0;
    case NV4097_SET_SURFACE_COLOR_DOFFSET:
        state->surface_color_offset[3] = data;
        state->surface_dirty = 1;
        return 0;
    case NV4097_SET_SURFACE_ZETA_OFFSET:
        state->surface_zeta_offset = data;
        state->surface_dirty = 1;
        return 0;
    case NV4097_SET_SURFACE_COLOR_TARGET:
        { static int _ct=0; if (_ct++ < 24 && getenv("RTT_DUMP"))
            fprintf(stderr, "[RSXCT] color_target=0x%X offA=0x%X offB=0x%X offC=0x%X offD=0x%X%s",
                    data, state->surface_color_offset[0], state->surface_color_offset[1],
                    state->surface_color_offset[2], state->surface_color_offset[3], "\n"); }
        state->color_target = data;
        state->surface_dirty = 1;
        return 0;
    case NV4097_SET_SURFACE_PITCH_A:
        state->surface_color_pitch[0] = data;
        return 0;
    case NV4097_SET_SURFACE_PITCH_B:
        state->surface_color_pitch[1] = data;
        return 0;
    case NV4097_SET_SURFACE_PITCH_C:
        state->surface_color_pitch[2] = data;
        return 0;
    case NV4097_SET_SURFACE_PITCH_D:
        state->surface_color_pitch[3] = data;
        return 0;
    case NV4097_SET_SURFACE_PITCH_Z:
        state->surface_zeta_pitch = data;
        return 0;
    default:
        return -1;
    }
}

/* ---------------------------------------------------------------------------
 * Texture method processing
 *
 * Texture registers are laid out in 16 units, each spanning 0x20 bytes:
 *   Unit 0: 0x1A00..0x1A1C
 *   Unit 1: 0x1A20..0x1A3C
 *   ...
 *   Unit N: 0x1A00 + N*0x20 .. 0x1A1C + N*0x20
 * -----------------------------------------------------------------------*/

static int process_texture_method(rsx_state* state, u32 method, u32 data)
{
    /* Compute texture unit index and register offset within the unit */
    u32 base = method - NV4097_SET_TEXTURE_OFFSET; /* 0x1A00 */
    u32 unit = base / 0x20;
    u32 reg  = base % 0x20;

    if (unit >= RSX_MAX_TEXTURES)
        return -1;

    rsx_texture_state* tex = &state->textures[unit];

    switch (reg) {
    case 0x00: /* TEXTURE_OFFSET */
        tex->offset = data;
        break;
    case 0x04: /* TEXTURE_FORMAT */
        tex->format = data;
        break;
    case 0x08: /* TEXTURE_ADDRESS (wrap S/T/R) */
        tex->address = data;
        break;
    case 0x0C: /* TEXTURE_CONTROL0 (enable, min/max LOD, max aniso) */
        tex->control0 = data;
        break;
    case 0x10: /* TEXTURE_CONTROL1 (remap) */
        tex->control1 = data;
        break;
    case 0x14: /* TEXTURE_FILTER (bias, min/mag filter) */
        tex->filter = data;
        break;
    case 0x18: /* TEXTURE_IMAGE_RECT (width << 16 | height) */
        tex->image_rect = data;
        break;
    case 0x1C: /* TEXTURE_BORDER_COLOR */
        tex->border_color = data;
        break;
    default:
        return -1;
    }

    tex->dirty = 1;
    state->texture_dirty = 1;
    return 0;
}

/* Vertex texture unit block: 0x0900 + unit*0x20, eight words. Same order as a
 * fragment unit's except at +0x10, which is CONTROL3 (the row pitch) rather
 * than CONTROL1 (the component crossbar) -- a vertex unit has no crossbar. */
static int process_vertex_texture_method(rsx_state* state, u32 method, u32 data)
{
    u32 base = method - NV4097_SET_VERTEX_TEXTURE_OFFSET;
    u32 unit = base / 0x20;
    u32 reg  = base % 0x20;

    if (unit >= RSX_MAX_VERTEX_TEXTURES)
        return -1;

    rsx_texture_state* tex = &state->vertex_textures[unit];

    switch (reg) {
    case 0x00: tex->offset       = data; break;
    case 0x04: tex->format       = data; break;
    case 0x08: tex->address      = data; break;
    case 0x0C: tex->control0     = data; break;
    case 0x10: tex->control3     = data; break;   /* CONTROL1's slot */
    case 0x14: tex->filter       = data; break;
    case 0x18: tex->image_rect   = data; break;
    case 0x1C: tex->border_color = data; break;
    default:
        return -1;
    }

    tex->dirty = 1;
    state->texture_dirty = 1;
    return 0;
}

/* SET_TEXTURE_CONTROL3: 0x1840 + unit*4, one word per unit rather than a slot
 * in the 0x20-byte unit block. */
static int process_texture_control3(rsx_state* state, u32 method, u32 data)
{
    u32 unit = (method - NV4097_SET_TEXTURE_CONTROL3) / 4;
    if (unit >= RSX_MAX_TEXTURES)
        return -1;
    state->textures[unit].control3 = data;
    state->textures[unit].dirty = 1;
    state->texture_dirty = 1;
    return 0;
}

/* ---------------------------------------------------------------------------
 * Vertex attribute method processing
 *
 * FORMAT registers: 0x1740 + attrib*4  (16 attributes)
 * OFFSET registers: 0x1680 + attrib*4  (16 attributes)
 * -----------------------------------------------------------------------*/

static int process_vertex_attrib_method(rsx_state* state, u32 method, u32 data)
{
    if (method >= NV4097_SET_VERTEX_DATA_ARRAY_FORMAT &&
        method < NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + RSX_MAX_VERTEX_ATTRIBS * 4) {
        u32 index = (method - NV4097_SET_VERTEX_DATA_ARRAY_FORMAT) / 4;
        rsx_vertex_attrib* attr = &state->vertex_attribs[index];

        /*
         * Format register layout:
         *   [3:0]   type   (float, half, ubyte, short, etc.)
         *   [7:4]   size   (number of components: 1-4)
         *   [15:8]  stride (bytes between consecutive elements)
         *   [16]    enable
         */
        attr->type      = data & 0xF;
        attr->size      = (data >> 4) & 0xF;
        attr->stride    = (data >> 8) & 0xFF;
        attr->frequency = (data >> 16) & 0xFFFF;   /* instancing divisor */
        /* SIZE (the component count) is what enables the attribute: NV4097 uses
         * size == 0 to mean "this array is off". Keying off `type` instead marked
         * every unused slot enabled-with-zero-components, because PSGL writes a
         * bare type (e.g. 0x00000002 = float32, size 0, stride 0) into the slots
         * it is NOT using -- so the input layout was built from 16 attributes of
         * which most were degenerate, and the draws rasterized nothing. */
        attr->enabled   = (attr->size != 0);
        attr->format    = data;
        state->vertex_dirty = 1;
        return 0;
    }

    if (method >= NV4097_SET_VERTEX_DATA_ARRAY_OFFSET &&
        method < NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + RSX_MAX_VERTEX_ATTRIBS * 4) {
        u32 index = (method - NV4097_SET_VERTEX_DATA_ARRAY_OFFSET) / 4;
        state->vertex_attribs[index].offset = data;
        state->vertex_dirty = 1;
        return 0;
    }

    return -1;
}

/* ---------------------------------------------------------------------------
 * Main method dispatch
 * -----------------------------------------------------------------------*/

/* Primitive batches (BEGIN_END begins) since the last flip; read and reset by
 * cellGcmSys' flip path for FRAME_DRAWS=1 (a renderer-independent "how much
 * is the guest drawing" -- the Linux null backend has no title bar). */
volatile unsigned g_rsx_frame_draws = 0;
/* v64: bumped whenever the vertex-program microcode store actually CHANGES (re-uploading the
 * same program -- every draw in inFamous -- does not); the backend caches its per-draw VP
 * hashing on it. */
volatile u32 g_rsx_vp_gen = 1;
volatile unsigned g_rsx_frame_verts = 0;  /* FRAME_DRAWS: vertices (indices) submitted this frame */
volatile unsigned g_rsx_frame_xverts = 0; /* ... expanded to triangle lists (what the D3D12 VP path stores) */
volatile unsigned long long g_rsx_frame_pbytes = 0; /* ... x 16 B x (highest enabled attrib + 1) */
static void frame_vert_account(const rsx_state* st, u32 count)
{
    u32 x = count;
    switch (st->primitive_type) {
    case RSX_PRIMITIVE_QUADS: x = count / 4 * 6; break;
    case RSX_PRIMITIVE_TRIANGLE_STRIP: case RSX_PRIMITIVE_TRIANGLE_FAN: x = count >= 3 ? 3 * (count - 2) : 0; break;
    default: break;
    }
    int hi = 0;
    for (int i = 0; i < 16; i++) if (st->vertex_attribs[i].enabled) hi = i;
    g_rsx_frame_verts += count;
    g_rsx_frame_xverts += x;
    g_rsx_frame_pbytes += (unsigned long long)x * 16u * (unsigned)(hi + 1);
}


/* v33 STENCIL_LOG=<n>: the first n distinct (fp, depth, stencil) states at draw time. */
static void rsx_stencil_log(const rsx_state* st)
{
    static int cap = -1; static int n = 0;
    if (cap < 0) { const char* e = getenv("STENCIL_LOG"); cap = e ? atoi(e) : 0; }
    if (!cap || n >= cap) return;
    { static int f = -1; static u32 lo = 0, hi = 0;
      if (f < 0) { const char* e = getenv("STENCIL_FP"); f = e ? 1 : 0; if (e) { lo = (u32)strtoul(e, NULL, 16); const char* c = strchr(e, '-'); hi = c ? (u32)strtoul(c + 1, NULL, 16) : lo; } }
      if (f && (st->shader_program < lo || st->shader_program > hi)) return; }
    { static int sf = -1; static u32 surf = 0;
      if (sf < 0) { const char* e = getenv("STENCIL_SURF"); sf = e ? 1 : 0; if (e) surf = (u32)strtoul(e, NULL, 16); }
      if (sf && (st->surface_color_offset[0] != surf || !st->depth_mask)) return; }
    static u32 seen[512][4]; static int ns = 0;
    u32 k0 = st->shader_program, k1 = (st->stencil_test_enable << 31) | (st->stencil2_enable << 30) | ((st->stencil_func & 0xFFF) << 16) | ((st->stencil_ref & 0xFF) << 8) | (st->stencil_mask & 0xFF);
    u32 k2 = ((st->stencil_op_fail & 0xFFF) << 20) ^ ((st->stencil_op_zfail & 0xFFF) << 8) ^ (st->stencil_op_zpass & 0xFFF) ^ (st->stencil_wmask << 24);
    u32 k3 = (st->depth_test_enable << 31) | (st->depth_mask << 30) | (st->depth_func & 0xFFFF);
    for (int i = 0; i < ns; i++) if (seen[i][0] == k0 && seen[i][1] == k1 && seen[i][2] == k2 && seen[i][3] == k3) return;
    if (ns < 512) { seen[ns][0] = k0; seen[ns][1] = k1; seen[ns][2] = k2; seen[ns][3] = k3; ns++; }
    n++;
    fprintf(stderr, "[BLENDST] en=%d sf=0x%X df=0x%X eq=0x%X col=0x%X cmask=0x%X\n", st->blend_enable, st->blend_sfactor, st->blend_dfactor, st->blend_equation, st->blend_color, st->color_mask);
    fprintf(stderr, "[STENCIL] fp=0x%08X ct=0x%X surf=0x%X zeta=0x%X | z test=%d write=%d func=0x%X | s en=%d 2s=%d func=0x%X ref=0x%X fmask=0x%X wmask=0x%X ops=%X/%X/%X | back func=0x%X ref=0x%X ops=%X/%X/%X\n",
            st->shader_program, st->color_target, st->surface_color_offset[0], st->surface_zeta_offset,
            st->depth_test_enable, st->depth_mask, st->depth_func,
            st->stencil_test_enable, st->stencil2_enable, st->stencil_func, st->stencil_ref, st->stencil_mask, st->stencil_wmask,
            st->stencil_op_fail, st->stencil_op_zfail, st->stencil_op_zpass,
            st->bstencil_func, st->bstencil_ref, st->bstencil_op_fail, st->bstencil_op_zfail, st->bstencil_op_zpass);
}
int rsx_process_method(rsx_state* state, u32 method, u32 data)
{
    /* RSX_MHIST=<k>[:<len_k>]: after <k> thousand methods, count the next len_k thousand (default 4000) and
     * print a histogram (method offset, count, last data) -- what a frame is made of. */
    { static long long _mh=-2, _ml=4000000LL; static long long _mc=0; static unsigned _cnt[0x4000]; static unsigned _last[0x4000];
      if(_mh==-2){ const char* e=getenv("RSX_MHIST"); _mh = e ? atoll(e)*1000LL : -1; if(e&&strchr(e,':')) _ml=atoll(strchr(e,':')+1)*1000LL; }
      if(_mh>=0){ _mc++; if((_mc%250000)==0) fprintf(stderr,"[mhist] n=%lld\n",_mc); if(_mc>_mh && _mc<=_mh+_ml){ unsigned k=(method>>2)&0x3FFF; _cnt[k]++; _last[k]=data; }
        if(_mc==_mh+_ml){ fprintf(stderr,"[mhist] after %lld methods:\n", _mc);
          for(unsigned k=0;k<0x4000;k++) if(_cnt[k]) fprintf(stderr,"[mhist] 0x%04X %8u last=0x%08X\n", k<<2, _cnt[k], _last[k]); } } }
    /* RSX_TRACE=<N>: trace the first N methods (bare "1" keeps the old 250).
     * The fixed 250 was spent entirely on boot-time setup, so the methods around
     * the first real draw -- exactly the ones worth seeing -- were never traced. */
    { static int _rt=-1; if(_rt<0){ const char* e=getenv("RSX_TRACE");
        _rt = e ? (atoi(e) > 1 ? atoi(e) : 250) : 0; }
      if(_rt){ static int _m=0; if(_m++<_rt) fprintf(stderr,"[rsxm] method=0x%04X data=0x%08X\n", method, data); } }
    /* IMM_MTRACE=<N>: trace the first N methods issued INSIDE a BEGIN/END pair
     * (immediate-mode vertex data order matters: attribute 0 provokes the vertex). */
    { static int _it=-1; if(_it<0){ const char* e=getenv("IMM_MTRACE"); _it = e ? atoi(e) : 0; }
      if(_it && state->in_begin_end){ static int _m=0; if(_m++<_it) fprintf(stderr,"[immm] method=0x%04X data=0x%08X\n", method, data); } }
    /* Back-end write label / semaphore (cellGcmSetWriteBackEndLabel): the RSX
     * writes a value to a report/label the CPU polls for CPU<->RSX sync (double
     * buffering). Real hardware DOES this; without it the game's frame-fence
     * loop (func_0006E6E0 polling label 0x41 @ 0x03000410) spins forever and
     * never reaches render. NV4097_SET_SEMAPHORE_OFFSET(0x1D6C)=offset (index*0x10
     * into the GCM label window @0x03000000); NV4097_BACK_END_WRITE_SEMAPHORE_
     * RELEASE(0x1D70)=value. Also NV406E semaphore release (0x0010 offset/0x0014
     * value) for the sub-channel path. */
    /* NV4097_SET_FREQUENCY_DIVIDER_OPERATION: per-attribute modulo/divide mask
     * for the frequency divisor (instancing). Captured for read_vp_vertex. */
    if (method == 0x00001FC0) { state->frequency_divider_op = data; return 0; }
    if (method == 0x00001FEC) {   /* LX: SET_SHADER_PACKER (sRGB output enable) */
        if (state) {
            static unsigned long n_on = 0, n_tot = 0; static int nl = 0;
            n_tot++; if (data & 1u) n_on++;
            if (nl < 12 || (n_tot % 100000ul) == 0) { nl++;
                fprintf(stderr, "[PACKER] SET_SHADER_PACKER = 0x%X (%lu of %lu writes enable sRGB output)\n", data, n_on, n_tot); }
            state->shader_packer = data;
        }
        return 0;
    }

    { static u32 s_sem_off = 0;
      if (method == 0x1D6C) { s_sem_off = data; return 0; }
      if (method == 0x1D70) {
        extern void vm_write32(uint32_t a, uint32_t v);
        if (g_rsx_vtx_sync) g_rsx_vtx_sync();   /* v33: the guest may reuse vertex memory after this label */
        /* The RSX back-end semaphore write swaps bytes 0<->2 of the value (hw
         * quirk); libgcm's cellGcmSetWriteBackEndLabel PRE-swaps to compensate
         * (SDK gcm_implementation_sub.h: "// swap byte 0 and 2"). Writing the
         * FIFO value verbatim left the label byte-swapped, so gcmutil's
         * cellGcmUtilFinish poll (label 255 == sLabelVal, starts at 1) compared
         * 0x00010000 != 1 and spun forever (gcm/cube sample hang after init).
         * Apply the same swap the hardware does so the pre-swap cancels out. */
        u32 val = (data & 0xff00ff00u) | ((data >> 16) & 0xffu) | ((data & 0xffu) << 16);
        vm_write32(VM_HLE_INJECT_BASE + (s_sem_off & 0x00FFFFFFu), val);
        { static int _l=0; if(_l++<8) fprintf(stderr,"[RSX] label write @0x%08X = 0x%08X (sync fence, raw 0x%08X)\n", VM_HLE_INJECT_BASE+(s_sem_off&0xFFFFFF), val, data); }
        return 0;
      }
      if (method == 0x1D74) {
        extern void vm_write32(uint32_t a, uint32_t v);
        if (g_rsx_vtx_sync) g_rsx_vtx_sync();
        vm_write32(VM_HLE_INJECT_BASE + (s_sem_off & 0x00FFFFFFu), data);
        return 0;
      } }
    if ((method >= 0x200 && method <= 0x23C) ||
        (method >= 0x280 && method <= 0x28C))
        return process_surface_method(state, method, data);

    /* Texture methods: 0x1A00..0x1A00 + 16*0x20 - 1 */
    if (method >= 0x1A00 && method < 0x1A00 + RSX_MAX_TEXTURES * 0x20)
        return process_texture_method(state, method, data);

    /* Texture CONTROL3: 0x1840..0x187C */
    if (method >= NV4097_SET_TEXTURE_CONTROL3 &&
        method < NV4097_SET_TEXTURE_CONTROL3 + RSX_MAX_TEXTURES * 4)
        return process_texture_control3(state, method, data);

    /* Vertex texture units: 0x0900..0x097C */
    if (method >= NV4097_SET_VERTEX_TEXTURE_OFFSET &&
        method < NV4097_SET_VERTEX_TEXTURE_OFFSET + RSX_MAX_VERTEX_TEXTURES * 0x20)
        return process_vertex_texture_method(state, method, data);

    /* Vertex attribute FORMAT: 0x1740..0x177C */
    if (method >= 0x1740 && method < 0x1740 + RSX_MAX_VERTEX_ATTRIBS * 4)
        return process_vertex_attrib_method(state, method, data);

    /* Vertex attribute OFFSET: 0x1680..0x16BC */
    if (method >= 0x1680 && method < 0x1680 + RSX_MAX_VERTEX_ATTRIBS * 4)
        return process_vertex_attrib_method(state, method, data);

    /* Viewport */
    /* Viewport transform: window = ndc*scale + offset. The z lane is the
     * GL->[0,1] depth remap (offset.z=0.5/scale.z=0.5); without honoring it,
     * GL-convention projections (SDK gcm samples) get their near-camera
     * geometry clipped by D3D's 0<=z<=w rule (gcm/cube: missing polygons). */
    if (method >= 0x0A20 && method < 0x0A30) {
        u32 f = data; float v; memcpy(&v, &f, 4);
        state->viewport_offset[(method - 0x0A20) >> 2] = v;
        return 0;
    }
    if (method >= 0x0A30 && method < 0x0A40) {
        u32 f = data; float v; memcpy(&v, &f, 4);
        state->viewport_scale[(method - 0x0A30) >> 2] = v;
        return 0;
    }

    if (method == NV4097_SET_VIEWPORT_HORIZONTAL) {
        state->viewport_x = data & 0xFFFF;
        state->viewport_w = (data >> 16) & 0xFFFF;
        state->viewport_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_VIEWPORT_VERTICAL) {
        state->viewport_y = data & 0xFFFF;
        state->viewport_h = (data >> 16) & 0xFFFF;
        state->viewport_dirty = 1;
        return 0;
    }

    /* v39: NV4097_SET_ANTI_ALIASING_CONTROL (0x1D7C): bit 0 MSAA enable, bit 4 ALPHA-TO-COVERAGE,
     * bit 8 alpha-to-one, bits 16-31 sample mask. Titles fade far-LOD meshes with alpha-to-coverage;
     * ignoring it drew inFamous' whole-world LOD pass (alpha 0 = invisible on hardware) over the lit
     * frame. Kept in a global (read by the D3D12 backend at record time; RSX thread only) so
     * rsx_commands.h -- and every file including it -- does not change. */
    /* v160R: NV4097_SET_SHADER_WINDOW (0x1D88): bits 0-11 height, 12-15 window origin
     * (0 top, 1 bottom), 16-19 pixel centre.  Not applied by the D3D12 backend; kept in a
     * global for the [CULLVP] diagnostics (was "[RSX-UNK] 0x1D88"). */
    if (method == 0x1D88) {
        extern u32 g_rsx_shader_window;
        g_rsx_shader_window = data;
        { static u32 seen[8]; static int ns = 0; int k = 0;
          for (int i = 0; i < ns; i++) if (seen[i] == data) { k = 1; break; }
          if (!k && ns < 8) { seen[ns++] = data;
              fprintf(stderr, "[SHADERWIN] v160R SHADER_WINDOW = 0x%08X (height %u, origin %s, pixel centre %u; fp 0x%X)\n",
                      data, data & 0xFFFu, ((data >> 12) & 0xFu) ? "bottom" : "top", (data >> 16) & 0xFu, state->shader_program); } }
        return 0;
    }
    if (method == 0x1D7C) {
        extern u32 g_rsx_aa_control;
        g_rsx_aa_control = data;
        { static u32 seen[32]; static int ns = 0; int k = 0;
          for (int i = 0; i < ns; i++) if (seen[i] == data) { k = 1; break; }
          if (!k && ns < 32) { seen[ns++] = data;
              fprintf(stderr, "[AA] ANTI_ALIASING_CONTROL = 0x%08X (msaa %u, alpha-to-coverage %u, alpha-to-one %u, sample mask 0x%04X; fp 0x%X)\n",
                      data, data & 1u, (data >> 4) & 1u, (data >> 8) & 1u, data >> 16, state->shader_program); } }
        return 0;
    }

    /* v90: MRT colour masks / MRT blend enables (were "[RSX-UNK]"). Decals that blend into
     * the G-buffer mask the normal target's ALPHA (material data) off; without the mask
     * they wrote alpha 0 around every leaf/garbage decal -> black squares after lighting. */
    if (method == 0x0370) { state->color_mask_mrt = data; state->mrt_seen |= 1; state->color_mask_dirty = 1;
        { static u32 seen[16]; static int ns = 0; int k = 0; for (int i = 0; i < ns; i++) if (seen[i] == data) k = 1;
          if (!k && ns < 16) { seen[ns++] = data; fprintf(stderr, "[MRT] colour mask MRT = 0x%08X (fp 0x%X)\n", data, state->shader_program); } }
        return 0; }
    if (method == 0x036C) { state->blend_enable_mrt = data; state->mrt_seen |= 2; state->blend_dirty = 1;
        { static u32 seen[16]; static int ns = 0; int k = 0; for (int i = 0; i < ns; i++) if (seen[i] == data) k = 1;
          if (!k && ns < 16) { seen[ns++] = data; fprintf(stderr, "[MRT] blend enable MRT = 0x%08X (RT0 blend %d, fp 0x%X)\n", data, state->blend_enable, state->shader_program); } }
        return 0; }
    /* Color mask */
    if (method == NV4097_SET_COLOR_MASK) {
        state->color_mask = data;
        state->color_mask_dirty = 1;
        return 0;
    }

    /* Alpha test */
    if (method == NV4097_SET_ALPHA_TEST_ENABLE) {
        state->alpha_test_enable = data ? 1 : 0;
        state->alpha_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_ALPHA_FUNC) {
        state->alpha_func = data;
        state->alpha_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_ALPHA_REF) {
        state->alpha_ref = data;
        state->alpha_dirty = 1;
        return 0;
    }

    /* Clear */
    if (method == NV4097_SET_COLOR_CLEAR_VALUE) {
        state->color_clear_value = data;
        return 0;
    }
    if (method == NV4097_SET_ZSTENCIL_CLEAR_VALUE) {
        state->zstencil_clear_value = data;
        return 0;
    }
    if (method == NV4097_CLEAR_SURFACE) {
        { static int _c=0; if (_c++ < 12) fprintf(stderr, "[RSX] CLEAR_SURFACE mask=0x%X color=0x%08X\n", data, state->color_clear_value); }
        { static long cl = -2; static long n = 0;   /* v33 CLEAR_LOG=<skip>: log clears after <skip> clears */
          if (cl == -2) { const char* e = getenv("CLEAR_LOG"); cl = e ? atol(e) : -1; }
          if (cl >= 0 && n++ >= cl && n < cl + 60000)
              fprintf(stderr, "[CLR] #%ld mask=0x%X surf=0x%X ct=0x%X zeta=0x%X color=0x%08X zs=0x%08X\n", n, data,
                      state->surface_color_offset[0], state->color_target, state->surface_zeta_offset,
                      state->color_clear_value, state->zstencil_clear_value); }
        if (s_backend && s_backend->clear) {
            float depth = (float)(state->zstencil_clear_value >> 8) / (float)0xFFFFFF;
            u8 stencil = state->zstencil_clear_value & 0xFF;
            s_backend->clear(s_backend->userdata, data,
                           state->color_clear_value, depth, stencil);
        }
        return 0;
    }

    /* Blend */
    if (method == NV4097_SET_BLEND_ENABLE) {
        state->blend_enable = data ? 1 : 0;
        state->blend_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_BLEND_FUNC_SFACTOR) {
        state->blend_sfactor = data;
        state->blend_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_BLEND_FUNC_DFACTOR) {
        state->blend_dfactor = data;
        state->blend_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_BLEND_EQUATION) {
        state->blend_equation = data;
        state->blend_dirty = 1;
        return 0;
    }

    /* Depth */
    if (method == NV4097_SET_DEPTH_TEST_ENABLE) {
        state->depth_test_enable = data ? 1 : 0;
        state->depth_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_DEPTH_FUNC) {
        state->depth_func = data;
        state->depth_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_DEPTH_MASK) {
        state->depth_mask = data ? 1 : 0;
        state->depth_dirty = 1;
        return 0;
    }

    /* Stencil */
    /* v33: depth range (cellGcmSetViewport min/max). Skyboxes / far-LOD passes use
     * min = max = 1 so they only land where the depth buffer is still clear. */
    if (method == 0x0394 || method == 0x0398) {
        float f; memcpy(&f, &data, 4);
        if (method == 0x0394) state->clip_min = f; else state->clip_max = f;
        if (method == 0x0398) { static float seen[16][2]; static int ns = 0; int k = 0;
            for (int i = 0; i < ns; i++) if (seen[i][0] == state->clip_min && seen[i][1] == state->clip_max) k = 1;
            if (!k && ns < 16) { seen[ns][0] = state->clip_min; seen[ns][1] = state->clip_max; ns++;
                fprintf(stderr, "[RSX] depth range [%g, %g] (fp 0x%X)\n", state->clip_min, state->clip_max, state->shader_program); } }
        return 0;
    }
    /* v35: depth bounds test (inFamous masks its far-LOD city meshes with it: they are
     * drawn over the whole frame, coincident with the real geometry, and only survive
     * where the stored depth is far). */
    if (method == 0x0380 || method == 0x0384 || method == 0x0388) {
        if (method == 0x0380) state->depth_bounds_enable = data ? 1 : 0;
        else { float f; memcpy(&f, &data, 4);
               if (method == 0x0384) state->depth_bounds_min = f; else state->depth_bounds_max = f; }
        state->depth_dirty = 1;
        { static u32 seen[16][3]; static int ns = 0; int k = 0; u32 mn, mx;
          memcpy(&mn, &state->depth_bounds_min, 4); memcpy(&mx, &state->depth_bounds_max, 4);
          for (int i = 0; i < ns; i++) if (seen[i][0] == (u32)state->depth_bounds_enable && seen[i][1] == mn && seen[i][2] == mx) k = 1;
          if (!k && ns < 16) { seen[ns][0] = (u32)state->depth_bounds_enable; seen[ns][1] = mn; seen[ns][2] = mx; ns++;
              fprintf(stderr, "[RSX] depth bounds %s [%g, %g] (fp 0x%X)\n", state->depth_bounds_enable ? "ON" : "off",
                      state->depth_bounds_min, state->depth_bounds_max, state->shader_program); } }
        return 0;
    }
    /* v33: stencil write mask + two-sided (back face) stencil */
    if (method == 0x032C) { state->stencil_wmask = data; state->stencil_dirty = 1; return 0; }
    if (method >= 0x0348 && method <= 0x0364) {
        switch (method) {
        case 0x0348: state->stencil2_enable = data ? 1 : 0; break;
        case 0x034C: state->bstencil_wmask = data; break;
        case 0x0350: state->bstencil_func = data; break;
        case 0x0354: state->bstencil_ref = data; break;
        case 0x0358: state->bstencil_mask = data; break;
        case 0x035C: state->bstencil_op_fail = data; break;
        case 0x0360: state->bstencil_op_zfail = data; break;
        case 0x0364: state->bstencil_op_zpass = data; break;
        }
        state->stencil_dirty = 1; return 0;
    }
    if (method == NV4097_SET_STENCIL_TEST_ENABLE) {
        state->stencil_test_enable = data ? 1 : 0;
        state->stencil_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_STENCIL_FUNC) {
        state->stencil_func = data;
        state->stencil_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_STENCIL_FUNC_REF) {
        state->stencil_ref = data;
        state->stencil_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_STENCIL_FUNC_MASK) {
        state->stencil_mask = data;
        state->stencil_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_STENCIL_OP_FAIL) {
        state->stencil_op_fail = data;
        state->stencil_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_STENCIL_OP_ZFAIL) {
        state->stencil_op_zfail = data;
        state->stencil_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_STENCIL_OP_ZPASS) {
        state->stencil_op_zpass = data;
        state->stencil_dirty = 1;
        return 0;
    }

    /* Culling */
    if (method == NV4097_SET_CULL_FACE_ENABLE) {
        state->cull_face_enable = data ? 1 : 0;
        return 0;
    }
    if (method == NV4097_SET_CULL_FACE) {
        state->cull_face = data;
        return 0;
    }
    if (method == NV4097_SET_FRONT_FACE) {
        state->front_face = data;
        return 0;
    }

    /* Shader programs */
    if (method == NV4097_SET_SHADER_PROGRAM) {
        /*
         * Fragment program address register:
         *   [1:0]  location (0 = local, 1 = main)
         *   [31:2] offset (4-byte aligned)
         */
        state->shader_program = data;
        state->fragment_program_addr = data & ~0x3u;
        state->shader_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_TRANSFORM_PROGRAM_LOAD) {
        /* Vertex program load slot — instruction index (each = 16 bytes).
         * Following NV4097_SET_TRANSFORM_PROGRAM words fill from here. */
        state->transform_program_load = data;
        state->vp_ucode_write = data * 16;
        state->shader_dirty = 1;
        return 0;
    }

    if (method == NV4097_SET_TRANSFORM_PROGRAM_START) {
        if (state->transform_program_start != data) state->vp_dirty = 1;
        state->transform_program_start = data;
        state->shader_dirty = 1;
        return 0;
    }

    /* NV4097_SET_TRANSFORM_PROGRAM[0..31] — a run of 32-bit vertex-program
     * microcode words appended at the current write cursor. Capture them so the
     * backend can decompile the real VP (4 words = one NV40 instruction). */
    if (method >= NV4097_SET_TRANSFORM_PROGRAM &&
        method <  NV4097_SET_TRANSFORM_PROGRAM + 32 * 4) {
        u32 w = state->vp_ucode_write;
        if (w + 4 <= sizeof(state->vp_ucode)) {
            /* data is host-endian already; store as little-endian bytes. */
            { u32 old; memcpy(&old, &state->vp_ucode[w], 4);
              if (old != data || w + 4 > state->vp_ucode_bytes) g_rsx_vp_gen++; }   /* v64 */
            state->vp_ucode[w+0] = (u8)(data);
            state->vp_ucode[w+1] = (u8)(data >> 8);
            state->vp_ucode[w+2] = (u8)(data >> 16);
            state->vp_ucode[w+3] = (u8)(data >> 24);
            state->vp_ucode_write = w + 4;
            if (w + 4 > state->vp_ucode_bytes) state->vp_ucode_bytes = w + 4;
            state->vp_dirty = 1;
        }
        return 0;
    }
    if (method == NV4097_SET_VERTEX_ATTRIB_OUTPUT_MASK) {
        state->vertex_attrib_output_mask = data;
        return 0;
    }
    /* NV4097_SET_VERTEX_DATA4F_M (0x1C00): the constant/"current" value for each
     * vertex attribute, 16 attributes x 4 floats. The hardware feeds this to
     * every vertex when the attribute's ARRAY is disabled -- it is not zero.
     *
     * Rubber Ducky leaves attribute 3 (diffuse colour) disabled and sets it here
     * instead; its duck shader computes (lighting * col0) * texture + spec, so a
     * zero col0 multiplies the texture away and the ducks rendered as a dim grey
     * specular term only -- present in the framebuffer, invisible on screen.
     * SET_VERTEX_DATA2F/4UB/2S/4S are the other encodings of the same register
     * file; add them if a title needs them. */
    /* NV4097_SET_VERTEX_DATA2F_M (0x1880, 2 floats per attribute) and
     * SET_VERTEX_DATA4UB_M (0x1940, one packed word per attribute): the same
     * constant register file in other encodings. Inside BEGIN/END they are
     * immediate-mode vertex data, and writing attribute 0 completes a vertex. */
    if (method >= 0x1880u && method < 0x1880u + RSX_MAX_VERTEX_ATTRIBS * 8u) {
        u32 idx  = (method - 0x1880u) >> 3;
        u32 lane = ((method - 0x1880u) >> 2) & 1;
        float f; memcpy(&f, &data, 4);
        state->vertex_data4f[idx][lane] = f;
        if (lane == 1) { state->vertex_data4f[idx][2] = 0.0f; state->vertex_data4f[idx][3] = 1.0f; }
        if (idx == 0 && lane == 1 && state->in_begin_end && state->imm_count < RSX_IMM_MAX_VERTS)
            memcpy(state->imm_verts[state->imm_count++], state->vertex_data4f, sizeof(state->vertex_data4f));
        return 0;
    }
    /* v160R: NV4097_SET_VERTEX_DATA3F_M (0x1500, 3 floats per attribute, w = 1): the same
     * constant register file; inFamous 2 programs attribute 0 this way (was "[RSX-UNK]
     * 0x1500/0x1504/0x1508").  VDATA3F=0 ignores it again. */
    if (method >= 0x1500u && method < 0x1500u + RSX_MAX_VERTEX_ATTRIBS * 12u) {
        static int on = -1;
        if (on < 0) { const char* e = getenv("VDATA3F"); on = (e && *e == '0') ? 0 : 1; }
        if (!on) return 0;
        u32 rel = method - 0x1500u, idx = rel / 12u, lane = (rel % 12u) >> 2;
        float f; memcpy(&f, &data, 4);
        state->vertex_data4f[idx][lane] = f;
        if (lane == 2) state->vertex_data4f[idx][3] = 1.0f;
        if (idx == 0 && lane == 2 && state->in_begin_end && state->imm_count < RSX_IMM_MAX_VERTS)
            memcpy(state->imm_verts[state->imm_count++], state->vertex_data4f, sizeof(state->vertex_data4f));
        { static int _n = 0; if (_n++ < 3)
            fprintf(stderr, "[VDATA3F] v160R attr %u lane %u = %g (in begin/end %d)\n", idx, lane, f, state->in_begin_end); }
        return 0;
    }
    if (method >= 0x1940u && method < 0x1940u + RSX_MAX_VERTEX_ATTRIBS * 4u) {
        u32 idx = (method - 0x1940u) >> 2;
        /* Bytes in memory order are x,y,z,w; data arrives host-endian. */
        state->vertex_data4f[idx][0] = (float)((data >> 24) & 0xFF) / 255.0f;
        state->vertex_data4f[idx][1] = (float)((data >> 16) & 0xFF) / 255.0f;
        state->vertex_data4f[idx][2] = (float)((data >>  8) & 0xFF) / 255.0f;
        state->vertex_data4f[idx][3] = (float)( data        & 0xFF) / 255.0f;
        return 0;
    }
    if (method >= 0x1C00u && method < 0x1C00u + RSX_MAX_VERTEX_ATTRIBS * 16u) {
        u32 idx  = (method - 0x1C00u) >> 4;         /* attribute */
        u32 lane = ((method - 0x1C00u) >> 2) & 3;   /* x/y/z/w   */
        float f; memcpy(&f, &data, 4);
        state->vertex_data4f[idx][lane] = f;
        if (idx == 0 && lane == 3 && state->in_begin_end && state->imm_count < RSX_IMM_MAX_VERTS)
            memcpy(state->imm_verts[state->imm_count++], state->vertex_data4f, sizeof(state->vertex_data4f));
        { static int _d = -1; if (_d < 0) _d = getenv("VDATA_DBG") ? 1 : 0;
          static int _n = 0;
          if (_d && _n < 400) { _n++;
            fprintf(stderr, "[VDATA4F] attr=%u lane=%u = %.4f%c", idx, lane, f, 10); } }
        return 0;
    }

    if (method == NV4097_SET_TRANSFORM_CONSTANT_LOAD) {
        { static int _n=0, _e=-1; if (_e < 0) _e = getenv("LOAD_DBG") ? 1 : 0; if (_e && _n++ < 200)
            fprintf(stderr, "[LOAD] transform_constant_load = %u\n", data); }
        state->transform_constant_load = data;
        return 0;
    }

    /* NV4097_SET_TRANSFORM_CONSTANT[0..63] — up to 64 dwords (16 vec4s) per
     * command. Each register slot writes to a lane of one vertex constant
     * vec4: vec_index = LOAD + (reg_offset/4), lane = reg_offset%4.
     * The data arrives as a host-endian u32; reinterpret the bits as float
     * because the game's intent is "these 32 bits are a float". The hardware
     * does NOT auto-advance LOAD between commands — games re-issue
     * SET_TRANSFORM_CONSTANT_LOAD before each block. */
    if (method >= NV4097_SET_TRANSFORM_CONSTANT &&
        method <  NV4097_SET_TRANSFORM_CONSTANT + 64 * 4) {
        u32 reg_offset = (method - NV4097_SET_TRANSFORM_CONSTANT) / 4;
        u32 slot = state->transform_constant_load + (reg_offset >> 2);
        u32 lane = reg_offset & 3;
        if (slot < RSX_MAX_VERTEX_CONSTANTS) {
            float f;
            memcpy(&f, &data, 4);
            { static int _en=-1; if(_en<0){const char*e=getenv("TCONST_DBG");_en=e?1:0;}
              static int _n=0;
              static int _max=-1; if(_max<0){const char*m=getenv("TCONST_MAX");_max=m?atoi(m):64;}
              int _hit = _en && (getenv("TCONST_ALL") ? (_n<_max) : (slot>=12 && slot<=30 && _n<400));
              if(_hit){ _n++; fprintf(stderr,"[TCONST] load=%u slot=%u lane=%u = %.4f\n", state->transform_constant_load, slot, lane, f); } }
            state->vertex_constants[slot][lane] = f;
            { static int _sq=0, _e=-1; if (_e < 0) _e = getenv("SEQ_DBG") ? 1 : 0; if (_e && slot==256 && lane==0 && _sq++ < 500)
                fprintf(stderr, "[SEQ] upload c256.x=%.4f (load=%u)\n", f, state->transform_constant_load); }
            if (!state->vertex_constants_dirty) {
                state->vertex_constants_lo = slot;
                state->vertex_constants_hi = slot;
                state->vertex_constants_dirty = 1;
            } else {
                if (slot < state->vertex_constants_lo) state->vertex_constants_lo = slot;
                if (slot > state->vertex_constants_hi) state->vertex_constants_hi = slot;
            }
        }
        return 0;
    }

    /* Draw */
    if (method == NV4097_SET_BEGIN_END) {
        if (data != 0) {
            state->primitive_type = data;
            state->in_begin_end = 1;
            state->begin_epoch++;
            { extern volatile unsigned g_rsx_frame_draws; g_rsx_frame_draws++; }
            state->imm_count = 0;
            /* v32 FP_DUMP=<dir>: every distinct fragment program's raw ucode (4 KB) and
             * its decompiled HLSL, named by address + colour target, for offline
             * shader debugging (works with the null backend). */
            { static int s_fd = -1; static char s_dir[256];
              if (s_fd < 0) { const char* e = getenv("FP_DUMP"); s_fd = (e && *e) ? 1 : 0;
                              if (s_fd) snprintf(s_dir, sizeof s_dir, "%s", e); }
              if (s_fd) {
                  static u32 tseen[16];
                  for (int ai = 0; ai < 16; ai++) {
                      const rsx_vertex_attrib* va = &state->vertex_attribs[ai];
                      if (!va->enabled || !va->stride) continue;
                      if (!(tseen[ai] & (1u << (va->type & 31)))) { tseen[ai] |= 1u << (va->type & 31);
                          fprintf(stderr, "[VTXTYPES] attr %d type %u size %u stride %u (fp 0x%08X)\n",
                                  ai, va->type, va->size, va->stride, state->shader_program); }
                  }
                  static u32 seen[1024]; static unsigned nseen = 0;
                  u32 fpa = state->shader_program; unsigned i;
                  for (i = 0; i < nseen; i++) if (seen[i] == fpa) break;
                  if (i == nseen && nseen < 1024) {
                      seen[nseen++] = fpa;
                      extern u32 cellGcmResolveLocated(int local, u32 offset);
                      extern uint8_t* vm_base;
                      extern int rsx_fp_decompile_ex(const u8*, u32, u32, u32, char*, u32);
                      u32 off = cellGcmResolveLocated((fpa & 3u) == 1, fpa & ~3u);
                      char path[400];
                      snprintf(path, sizeof path, "%s/fp_%08X_ct%02X.bin", s_dir, fpa, state->color_target);
                      FILE* f = fopen(path, "wb");
                      if (f) { fwrite(vm_base + off, 1, 4096, f); fclose(f); }
                      static char hl[65536];
                      int n = rsx_fp_decompile_ex(vm_base + off, 4096, state->shader_control & 0x40u, 0u, hl, sizeof hl);
                      snprintf(path, sizeof path, "%s/fp_%08X_ct%02X.hlsl", s_dir, fpa, state->color_target);
                      f = fopen(path, "w");
                      if (f) { fprintf(f, "// fp=0x%08X ctrl=0x%X color_target=0x%X n=%d\n%s\n", fpa, state->shader_control,
                                       state->color_target, n, n > 0 ? hl : "(decompile failed)"); fclose(f); }
                  }
              } }

            /* Flush dirty state to backend before drawing */
            if (s_backend) {
                if (state->surface_dirty && s_backend->set_render_target)
                    s_backend->set_render_target(s_backend->userdata, state);
                if (state->viewport_dirty && s_backend->set_viewport)
                    s_backend->set_viewport(s_backend->userdata, state);
                if (state->blend_dirty && s_backend->set_blend)
                    s_backend->set_blend(s_backend->userdata, state);
                if ((state->depth_dirty || state->stencil_dirty) && s_backend->set_depth_stencil)
                    s_backend->set_depth_stencil(s_backend->userdata, state);
                if (state->color_mask_dirty && s_backend->set_color_mask)
                    s_backend->set_color_mask(s_backend->userdata, state);
                if (state->alpha_dirty && s_backend->set_alpha_test)
                    s_backend->set_alpha_test(s_backend->userdata, state);
                if (state->shader_dirty && s_backend->set_shader)
                    s_backend->set_shader(s_backend->userdata, state);
                if (state->vertex_dirty && s_backend->set_vertex_attribs)
                    s_backend->set_vertex_attribs(s_backend->userdata, state);

                /* Bind dirty textures */
                if (state->texture_dirty && s_backend->bind_texture) {
                    for (u32 i = 0; i < RSX_MAX_TEXTURES; i++) {
                        if (state->textures[i].dirty) {
                            s_backend->bind_texture(s_backend->userdata, i, &state->textures[i]);
                            state->textures[i].dirty = 0;
                        }
                    }
                }

                state->surface_dirty = 0;
                state->viewport_dirty = 0;
                state->blend_dirty = 0;
                state->depth_dirty = 0;
                state->stencil_dirty = 0;
                state->color_mask_dirty = 0;
                state->alpha_dirty = 0;
                state->shader_dirty = 0;
                state->vertex_dirty = 0;
                state->texture_dirty = 0;
            }
        } else {
            state->in_begin_end = 0;
            /* Immediate-mode vertices collected since BEGIN: draw them now, with
             * every vertex array switched off so the fetch reads imm_verts. */
            if (state->imm_count && s_backend && s_backend->draw_arrays) {
                static int s_imm_off = -1;
                if (s_imm_off < 0) s_imm_off = getenv("RSX_NO_IMMEDIATE") ? 1 : 0;
                if (!s_imm_off) {
                    static unsigned s_imm_n = 0;
                    if (s_imm_n++ < 8)
                        fprintf(stderr, "[RSX] immediate-mode draw prim=%u verts=%u\n",
                                state->primitive_type, state->imm_count);
                    int saved[RSX_MAX_VERTEX_ATTRIBS];
                    for (u32 i = 0; i < RSX_MAX_VERTEX_ATTRIBS; i++) {
                        saved[i] = state->vertex_attribs[i].enabled;
                        state->vertex_attribs[i].enabled = 0;
                    }
                    { static int _id = -1, _in = 0; if (_id < 0) _id = getenv("IMM_DBG") ? 1 : 0;
                      if (_id && _in < 12) { _in++;
                        for (u32 _v = 0; _v < state->imm_count; _v++) {
                            fprintf(stderr, "[IMM] draw prim=%u v%u:", state->primitive_type, _v);
                            for (int _a = 0; _a < 16; _a++) {
                                const float* q = state->imm_verts[_v][_a];
                                if (q[0] || q[1] || q[2] || (q[3] && q[3] != 1.0f))
                                    fprintf(stderr, " a%d=(%.3f,%.3f,%.3f,%.3f)", _a, q[0], q[1], q[2], q[3]);
                            }
                            fprintf(stderr, "\n");
                        } } }
                    state->imm_active = 1;
                    if (s_backend->set_vertex_attribs)
                        s_backend->set_vertex_attribs(s_backend->userdata, state);
                    s_backend->draw_arrays(s_backend->userdata, state->primitive_type, 0, state->imm_count);
                    state->imm_active = 0;
                    for (u32 i = 0; i < RSX_MAX_VERTEX_ATTRIBS; i++)
                        state->vertex_attribs[i].enabled = saved[i];
                    state->vertex_dirty = 1;
                }
            }
            state->imm_count = 0;
        }
        return 0;
    }

    if (method == NV4097_DRAW_ARRAYS) {
        u32 first = data & 0xFFFFFF;
        u32 count = ((data >> 24) & 0xFF) + 1;
        { static int _d=0; if (_d++ < 32) fprintf(stderr, "[RSX] DRAW_ARRAYS prim=%u first=%u count=%u\n", state->primitive_type, first, count); }
        { static int _sq=0, _e=-1; if (_e < 0) _e = getenv("SEQ_DBG") ? 1 : 0; if (_e && _sq++ < 500)
            fprintf(stderr, "[SEQ] DRAW surf0=0x%X c256.x=%.4f c257.y=%.4f\n",
                    state->surface_color_offset[0], state->vertex_constants[256][0],
                    state->vertex_constants[257][1]); }
        /* MVPDBG: dump every non-zero vertex-constant slot the FIRST time a
         * G-buffer draw (surf0=0xCC0000) is dispatched -- the true MVP the GPU
         * will use, read live (no snapshot/parity indirection). */
        static int s_mvp = -1; if (s_mvp < 0) s_mvp = getenv("MVPDBG") ? 1 : 0;
        if (s_mvp && state->surface_color_offset[0] == 0xCC0000) {
            static int _once = 0;
            if (!_once) { _once = 1;
                for (int s = 0; s < RSX_MAX_VERTEX_CONSTANTS; s++) {
                    float* v = state->vertex_constants[s];
                    if (v[0]||v[1]||v[2]||v[3])
                        fprintf(stderr, "[MVP] c[%3d] = %10.4f %10.4f %10.4f %10.4f\n",
                                s, v[0], v[1], v[2], v[3]);
                }
            }
        }
        frame_vert_account(state, count);
        ps3_ms("rsx:draw_arrays");
        if (s_backend && s_backend->draw_arrays)
            rsx_stencil_log(state);
            s_backend->draw_arrays(s_backend->userdata, state->primitive_type, first, count);
        return 0;
    }

    /* v33: fixed-function fog inputs. The NV40 fragment pipe hands a program that reads
     * FOGC float4(fogc, fog_factor(mode, params, fogc), 0, 0) -- the factor is computed by
     * fixed function from these two registers (RPCS3 fetch_fog_value). */
    if (method == 0x08CC || method == 0x08D0 || method == 0x08D4) {
        if (method == 0x08CC) state->fog_mode = data;
        else { float f; memcpy(&f, &data, 4); state->fog_param[(method - 0x08D0) >> 2] = f; }
        { static u32 seen[24][3]; static int ns = 0; int k = 0;
          u32 p0, p1; memcpy(&p0, &state->fog_param[0], 4); memcpy(&p1, &state->fog_param[1], 4);
          for (int i = 0; i < ns; i++) if (seen[i][0] == state->fog_mode && seen[i][1] == p0 && seen[i][2] == p1) k = 1;
          if (!k && ns < 24) { seen[ns][0] = state->fog_mode; seen[ns][1] = p0; seen[ns][2] = p1; ns++;
              fprintf(stderr, "[FOG] mode=0x%X p0=%g p1=%g%c", state->fog_mode, state->fog_param[0], state->fog_param[1], 10); } }
        return 0;
    }
    if (method == NV4097_SET_SHADER_CONTROL) {
        { static int _sc=0; if (_sc++ < 12 && getenv("RTT_DUMP"))
            fprintf(stderr, "[RSXSC] shader_control=0x%X\n", data); }
        state->shader_control = data;
        state->shader_dirty = 1;
        return 0;
    }
    if (method == NV4097_SET_INDEX_ARRAY_ADDRESS) {
        state->index_array_offset = data;
        return 0;
    }
    if (method == NV4097_SET_INDEX_ARRAY_DMA) {
        state->index_array_dma = data;
        return 0;
    }
    if (method == NV4097_DRAW_INDEX_ARRAY) {
        /* [23:0] first index, [31:24] count-1 (same packing as DRAW_ARRAYS). */
        u32 first = data & 0xFFFFFF;
        u32 count = ((data >> 24) & 0xFF) + 1;
        { static int _d=0; if (_d++ < 8) fprintf(stderr, "[RSX] DRAW_INDEX_ARRAY prim=%u first=%u count=%u idxoff=0x%X dma=0x%X\n", state->primitive_type, first, count, state->index_array_offset, state->index_array_dma); }
        frame_vert_account(state, count);
        ps3_ms("rsx:draw_indexed");
        if (s_backend && s_backend->draw_indexed)
            rsx_stencil_log(state);
            s_backend->draw_indexed(s_backend->userdata, state->primitive_type,
                                    first, count);
        return 0;
    }

    /* Scissor */
    if (method == NV4097_SET_SCISSOR_HORIZONTAL) {
        state->scissor_x = data & 0xFFFF;
        state->scissor_w = (data >> 16) & 0xFFFF;
        return 0;
    }
    if (method == NV4097_SET_SCISSOR_VERTICAL) {
        state->scissor_y = data & 0xFFFF;
        state->scissor_h = (data >> 16) & 0xFFFF;
        return 0;
    }

    /* NV406E_SET_REFERENCE (0x0050): the title writes a reference value it then
     * spins on (cellGcmFinish / cellGcmSetWaitLabel). Record the latest so the
     * GCM control register can report completion back to the guest and unblock
     * the spin. Class 0 (software) method, no subchannel state needed here. */
    if (method == 0x0050) {
        if (g_rsx_vtx_sync) g_rsx_vtx_sync();   /* v33 */
        g_rsx_last_reference = data;
        return 0;
    }

    /* GCM_FLIP_HEAD (0xE920/0xE924): immediate display flip from the FIFO.
     * The draw engine handles the actual present; suppress the unknown log. */
    if (method == 0xE920 || method == 0xE924) return 0;

    /* v36: every DISTINCT unparsed method once, in release builds too (first 96): what the
     * title programs that the backend silently ignores (depth bounds hid here for a week). */
    { static u32 seen[96]; static int ns = 0; int k = 0;
      for (int i = 0; i < ns; i++) if (seen[i] == method) { k = 1; break; }
      if (!k && ns < 96) { seen[ns++] = method;
          fprintf(stderr, "[RSX-UNK] method 0x%04X = 0x%08X (first time)\n", method, data); } }
    /* Unrecognized method -- log in debug builds */
#ifndef NDEBUG
    static int s_unknown_count = 0;
    if (s_unknown_count < 50) {
        printf("[RSX] unknown method 0x%04X = 0x%08X\n", method, data);
        s_unknown_count++;
    }
#endif

    return -1;
}

/* ---------------------------------------------------------------------------
 * Command buffer parsing
 * -----------------------------------------------------------------------*/

int rsx_process_command_buffer(rsx_state* state, const u32* buf, u32 size)
{
    int methods_processed = 0;
    u32 pos = 0;
    u32 count = size / 4; /* size in dwords */

    { static int _c=0; if (count && _c++ < 16) fprintf(stderr, "[RSX] process_cmd_buffer words=%u first_hdr=0x%08X\n", count, buf[0]); }
    while (pos < count) {
        u32 header = buf[pos++];
        u32 type = (header >> 29) & 0x7;

        if (type == 0 || type == 2) {
            /* Increasing or non-increasing method */
            u32 method = (header >> 2) & 0x7FF;
            method <<= 2; /* method addresses are dword-aligned */
            u32 num_data = (header >> 18) & 0x7FF;
            int increasing = (type == 0);

            for (u32 i = 0; i < num_data && pos < count; i++) {
                u32 data = buf[pos++];
                u32 m = increasing ? (method + i * 4) : method;
                rsx_process_method(state, m, data);
                methods_processed++;
            }
        } else if (type == 1) {
            /* Jump — change command buffer read position */
            /* In recomp context, this is handled by the caller */
            break;
        } else {
            /* Unknown type, skip */
            break;
        }
    }

    return methods_processed;
}

/* v39: last NV4097_SET_ANTI_ALIASING_CONTROL value (see the 0x1D7C handler). */
u32 g_rsx_aa_control = 0xFFFF0000u;
/* v160R: last NV4097_SET_SHADER_WINDOW value (see the 0x1D88 handler). */
u32 g_rsx_shader_window = 0u;
