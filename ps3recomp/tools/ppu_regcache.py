#!/usr/bin/env python3
"""v103 PPU register caching -- source-to-source pass over the lifted PPU TUs.

usage: python ppu_regcache.py <in.cpp> <out.cpp> [--stats]     (env PPU_RC_MINREF, default 4)

The lifter writes every guest register access as ctx->gpr[N] / ctx->fpr[N] / ctx->cr / ctx->xer.
Every guest store goes through a char-typed memcpy into vm_base, which may alias *ctx, so the
compiler must reload every register after each store and store every result back at once.
This pass keeps the GPRs/FPRs/CR/XER that a lifted function references >= MINREF times in locals
(rc_gN, rc_fN, rc_cr, rc_xer) and synchronises them with *ctx only where someone else can look:

  * sync points: every call that gets ctx (func_X(ctx), ps3_indirect_call, lv2_syscall,
    ps3_hle_call, DRAIN_TRAMPOLINE) or may read g_active_ctx (ps3_taskwait_*()), every `return;`
    and the closing brace. Before a sync: store back the registers that MAY be dirty there
    (line-level forward dataflow over the goto graph). After a call: reload the "eager" set E.
  * loads are lazy: a register is loaded from ctx at the start of the first line that reads it
    while its local is not known valid. Registers for which that is unsafe (dirty on one path,
    stale on another at a read or a sync) are moved into E (loaded at entry, reloaded after every
    call) until the analysis is consistent.
  * a function whose lines are not single complete statements (unbalanced braces, braceless
    if/else) or that calls anything outside the known helper set is copied unchanged.

VR registers, lr, ctr and the reservation stay in ctx. Output is deterministic; it is generated at
build time from the hand-patched recompiled/*.cpp (CMake PORT_PPU_REGCACHE).
Verification (cloud, 2026-10-08): differential harness, old vs new from identical random state,
every runtime call hashes the whole ctx and clobbers registers; 0 diffs on ~1,400 functions x 40
seeds with NaN-free inputs; planted missing-spill / missing-reload bugs are caught (428 / 369 of 678
functions). Known benign difference: with NaN inputs, the host may pick a different NaN payload
(the compiler is free to commute FP operands / fold float<->double round trips).
"""
import re, sys, os
MINREF = int(os.environ.get('PPU_RC_MINREF', '4'))

FN_START = re.compile(r'^void (func_\w+)\(ppu_context\* ctx\) \{\s*$')
REG = re.compile(r'ctx->(gpr|fpr)\[(\d+)\]|ctx->(cr|xer)\b')
# write = assignment (not ==), compound assignment, ++/--, or address taken
WRITE_AFTER = re.compile(r'\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)|\s*(?:\+\+|--)')
SYNC = re.compile(
    r'(?:(?:\bfunc_[0-9A-Fa-f]{8}\w*|\bps3_indirect_call|\blv2_syscall)\(ctx\);'
    r'|\bps3_hle_call\([^;()]*\bctx\);'
    r'|\bDRAIN_TRAMPOLINE\(ctx\);'
    r'|\bps3_taskwait_\w+\(\);)'
    r'(?:\s*DRAIN_TRAMPOLINE\(ctx\);)?')
RET = re.compile(r'\breturn;')
CALL = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
ALLOWED = set('''vm_read8 vm_read16 vm_read32 vm_read64 vm_write8 vm_write16 vm_write32 vm_write64
memcpy memset if for while switch return sizeof void PPU_FENCE ppc_rldicl ppc_rldicr ppc_rldic ppc_rldimi
ppc_rlwinm ppc_rlwimi ppc_sraw ppc_srad ppc_mulhd ppc_mulhdu ppc_rotl64 ppc_mask64
ppu_vldf4 ppu_vstf4 ppu_vldu4 ppu_vstu4 ppu_stvx16 ppu_fp_single ppu_fmul ppu_fadd ppu_fsub ppu_fdiv
ppu_fmadd_core ppu_frsp ppu_f2i32 ppu_f2i64 ppu_stwcx32 ppu_stdcx64 ppu_timebase_now ppu_dcbz128
ppu_atomic_add16 ppu_vbswap32 sqrtf sqrt fabs fabsf signbit __builtin_clz __builtin_clzll
DRAIN_TRAMPOLINE ps3_indirect_call lv2_syscall ps3_hle_call
ps3_taskwait_begin ps3_taskwait_spin ps3_taskwait_end'''.split())
COMMENT = re.compile(r'/\*.*?\*/')


def regname(m):
    if m.group(1):
        return ('g' if m.group(1) == 'gpr' else 'f') + m.group(2)
    return m.group(3)


def local(r):
    return {'cr': 'rc_cr', 'xer': 'rc_xer'}.get(r, 'rc_' + r)


def ctxref(r):
    if r in ('cr', 'xer'):
        return 'ctx->' + r
    return 'ctx->%s[%s]' % ('gpr' if r[0] == 'g' else 'fpr', r[1:])


def ctype(r):
    return {'g': 'uint64_t', 'f': 'double'}.get(r[0], 'uint32_t') if r not in ('cr', 'xer') else 'uint32_t'


def order(r):
    if r in ('cr', 'xer'):
        return (2, r)
    return (0 if r[0] == 'g' else 1, int(r[1:]))


LOC = re.compile(r'\brc_(g\d+|f\d+|cr|xer)\b')
EVT = re.compile('(' + SYNC.pattern + ')|' + RET.pattern)
GOTO = re.compile(r'\bgoto (loc_\w+);')
UNCOND_PREFIX = re.compile(r'^\s*(?:ctx->lr = 0x[0-9A-Fa-f]+;\s*)?$')
BRACELESS = re.compile(r'^\s*(?:(?:if|for|while|switch)\b.*\)\s*$|else\b|case\b|default\b)')
LABEL = re.compile(r'^\s*(loc_\w+):')


def transform_fn(name, body, stats):
    """body = list of lines between the opening and the closing brace (exclusive)."""
    refs = {}
    for l in body:
        code = COMMENT.sub('', l)
        for c in CALL.finditer(code):
            n = c.group(1)
            if n.startswith('func_') or n in ALLOWED:
                continue
            stats['skipped'] += 1
            stats.setdefault('skip_names', set()).add(name + ':' + n)
            return None
        # the dataflow below is line-based: every line must be a complete statement sequence
        if code.count('{') != code.count('}') or BRACELESS.match(code):
            stats['skipped'] += 1
            stats.setdefault('skip_names', set()).add(name + ':multiline')
            return None
        for m in REG.finditer(l):
            r = regname(m)
            refs[r] = refs.get(r, 0) + 1
    # only registers referenced often enough are worth a local: every cached register costs a load
    # at entry and a store at exit/sync, and with ~30 live locals x86 runs out of callee-saved
    # registers (the slow-path calls in the memory helpers) and parks them on the stack anyway.
    used = set(r for r, c in refs.items() if c >= MINREF)
    if not used:
        return None
    U = sorted(used, key=order)
    bit = {r: 1 << k for k, r in enumerate(U)}
    tl = []          # transformed lines (registers -> locals)
    for l in body:
        nl = REG.sub(lambda m: local(regname(m)) if regname(m) in used else m.group(0), l)
        chk = EVT.sub('', nl)
        chk = REG.sub('', chk)
        chk = re.sub(r'ctx->(?:vr\[\d+\]|(?:lr|ctr|fpscr|vscr|reserve_addr|reserve_value|reserve_valid)\b)', '', chk)
        chk = COMMENT.sub('', chk)
        if re.search(r'\bctx\b', chk):
            stats['skipped_ctx'] += 1
            stats.setdefault('skip_names', set()).add(name + ':ctx:' + l.strip()[:60])
            return None
        tl.append(nl)
    # per line: ordered events (pos, kind, mask): 'R' read, 'w' write, 's' sync, 'c' conditional
    # sync, 'r' return. A register is cached lazily: loaded from ctx at the start of the first line
    # that reads it while its local is not known valid (V), so a call only costs the reloads of the
    # E set (registers that must stay valid everywhere) instead of every cached register.
    evs = []; labels = {}; succ = []; rd_line = []; late = 0
    for i, nl in enumerate(tl):
        e = []
        for m in LOC.finditer(nl):
            mk = bit[m.group(1)]
            pre = nl[max(0, m.start() - 1):m.start()]
            wa = WRITE_AFTER.match(nl, m.end())
            if pre == '&' or wa:
                e.append((m.start(), 'w', mk))
            if pre == '&' or not wa or not re.match(r'\s*=(?!=)', nl[m.end():]):
                e.append((m.start(), 'R', mk))   # plain read, compound assignment, ++/--, &x
        for m in EVT.finditer(nl):
            k = 'r'
            if m.group(1):
                # a sync resets the state only when it runs unconditionally: nothing but an
                # `ctx->lr = imm;` in front of it on its line (else: `if (c) { ...; sync }` etc.)
                k = 's' if UNCOND_PREFIX.match(COMMENT.sub('', nl[:m.start()])) else 'c'
            e.append((m.start(), k, 0))
        e.sort(key=lambda t: (t[0], t[1] != 'R'))
        evs.append(e)
        r_all = 0; seen_sync = False
        for pos, k, mk in e:
            if k in ('s', 'c'): seen_sync = True
            elif k == 'R':
                if seen_sync: late |= mk     # read after a call on the same line: keep it eager
                else: r_all |= mk
        rd_line.append(r_all)
        lm = LABEL.match(nl)
        if lm: labels[lm.group(1)] = i
        succ.append(GOTO.findall(nl))
    n = len(tl)
    ALL = (1 << len(U)) - 1
    E = late
    it = 0
    while True:
        it += 1
        vin = [ALL] * (n + 1); din = [0] * (n + 1); reach = [False] * (n + 1)
        vin[0] = E; reach[0] = True
        conflict = 0
        def run(i, v, d, emit=None):
            need = rd_line[i] & ~v
            nonlocal conflict
            conflict |= need & d
            v |= need
            if emit is not None: emit.append(need & ~d)
            for pos, k, mk in evs[i]:
                if k == 'w': v |= mk; d |= mk
                elif k in ('s', 'c', 'r'):
                    conflict |= d & ~v
                    if emit is not None: emit.append(d)
                    if k == 's': v = E; d = 0
                    elif k == 'c': v &= E   # merge of taken (V=E, D=0) and not-taken paths
            return v, d
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not reach[i]: continue
                v, d = run(i, vin[i], din[i])
                for t in [i + 1] + [labels[x] for x in succ[i] if x in labels]:
                    nv = vin[t] & v if reach[t] else v
                    nd = din[t] | d
                    if not reach[t] or nv != vin[t] or nd != din[t]:
                        reach[t] = True; vin[t] = nv; din[t] = nd
                        if t <= i: changed = True
        # end of function: spill what may be dirty (must be valid)
        if reach[n]: conflict |= din[n] & ~vin[n]
        if conflict & ~E == 0: break
        E |= conflict
    def spill(d):
        return ''.join('%s = %s; ' % (ctxref(r), local(r)) for r in U if d & bit[r])
    def loads(m):
        return ''.join('%s = %s; ' % (local(r), ctxref(r)) for r in U if m & bit[r])
    out = []; nsync = 0; nst = 0; nld = 0
    for i, nl in enumerate(tl):
        if not reach[i]:
            out.append(nl); continue
        sp = []
        run(i, vin[i], din[i], sp)
        ld = loads(sp[0]); nld += ld.count(';')
        k = iter(sp[1:])
        def _s(m):
            d = next(k); s_ = spill(d)
            nonlocal nsync, nst
            nst += s_.count(';')
            if m.group(1):
                nsync += 1
                return '{ ' + s_ + m.group(0) + ' RC_RL }'
            return '{ ' + s_ + 'return; }'
        nl2 = EVT.sub(_s, nl)
        if ld:
            lm = LABEL.match(nl2)
            if lm:   # keep the label first: `loc_X: loads; rest`
                nl2 = nl2[:lm.end()] + ' ' + ld + nl2[lm.end():]
            else:
                nl2 = '        ' + ld + nl2.lstrip(' ')
        out.append(nl2)
    end = spill(din[n]) if reach[n] else ''
    nst += end.count(';')
    stats['fns'] += 1; stats['sync'] += nsync; stats['regs'] += len(U); stats['spills'] += nst
    stats['lazyloads'] = stats.get('lazyloads', 0) + nld
    stats['eager'] = stats.get('eager', 0) + bin(E).count('1')
    stats['iters'] = max(stats.get('iters', 0), it)
    rl = ' '.join('%s = %s;' % (local(r), ctxref(r)) for r in U if E & bit[r])
    decl = ''.join('    %s %s%s;\n' % (ctype(r), local(r), (' = ' + ctxref(r)) if E & bit[r] else ' = 0') for r in U)
    head = '#define RC_RL %s\n' % rl
    return head, decl, out, end


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, 'r', encoding='latin-1', newline='').read()
    lines = data.splitlines(True)
    stats = dict(fns=0, skipped=0, skipped_ctx=0, sync=0, regs=0, spills=0, total=0)
    o = []
    o.append('/* v103 ppu_regcache.py: register-cached copy of %s -- generated at build time, do not edit. */\n'
             % src.replace('\\', '/').split('/')[-1])
    i, n = 0, len(lines)
    while i < n:
        l = lines[i]
        m = FN_START.match(l.rstrip('\r\n'))
        if not m:
            o.append(l); i += 1; continue
        j = i + 1
        while j < n and not lines[j].startswith('}'):
            j += 1
        if j >= n:
            raise SystemExit('unterminated function at line %d' % (i + 1))
        stats['total'] += 1
        body = lines[i + 1:j]
        r = transform_fn(m.group(1), body, stats)
        if r is None:
            o.extend(lines[i:j + 1])
        else:
            head, decl, out, end = r
            o.append(head); o.append(l); o.append(decl); o.extend(out)
            o.append('    ' + end + '\n'); o.append(lines[j])
            o.append('#undef RC_RL\n')
        i = j + 1
    open(dst, 'w', encoding='latin-1', newline='').write(''.join(o))
    if '--stats' in sys.argv:
        sk = stats.pop('skip_names', set())
        print(src, stats, file=sys.stderr)
        for s in sorted(sk)[:40]:
            print('  skipped', s, file=sys.stderr)


if __name__ == '__main__':
    main()
