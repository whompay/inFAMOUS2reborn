#!/usr/bin/env python3
"""gen_spu_rc.py -- F181 (perf): build-time register-cached twin of ONE lifted SPU image.

Usage: gen_spu_rc.py <ps3recomp/tools/spu_regcache.py> <spu/<img>/spu_recomp.c> <out spu_recomp_rc.c>

Runs the toolkit's v95 spu_regcache pass (keep SPU registers in host locals; ctx->gpr exact at
every call/channel/transfer/return) on one image and writes the twin into the BUILD tree, so the
source tree is untouched and the twin can never go stale: CMake re-runs this whenever the lifted
spu_recomp.c or the pass changes (CMakeLists PORT_SPU_RC_GEN). The twin's constructor hands the
runtime its (original, rc) pairs; SPU_REGCACHE=0 at run time serves the original code only.
"""
import importlib.util, os, re, sys

def main():
    tool, src_path, out_path = sys.argv[1:4]
    spec = importlib.util.spec_from_file_location('spu_regcache', tool)
    rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
    src = open(src_path, encoding='utf-8', errors='surrogateescape').read()
    dst, nf = rc.xform_file(src)
    dst, _ = rc.twin(dst, src)
    dst = re.sub(r'^(#include "[./]*(?:runtime/spu/)?spu_helpers\.h"[^\n]*)$',
                 r'\1\n#include "spu_helpers_vec.h"', dst, count=1, flags=re.M)
    if 'spu_helpers_vec.h' not in dst:
        sys.exit('gen_spu_rc: no spu_helpers.h include in ' + src_path)
    # The twin lives in the build tree: name the image's own header by absolute path and the
    # helpers through the runtime/spu include directory the SPU TUs already carry.
    hdr = os.path.join(os.path.dirname(os.path.abspath(src_path)), 'spu_recomp.h').replace('\\', '/')
    dst = re.sub(r'^#include "spu_recomp\.h"', '#include "%s"' % hdr, dst, count=1, flags=re.M)
    dst = re.sub(r'^#include "[./]*(?:runtime/spu/)?spu_helpers\.h"', '#include "spu_helpers.h"', dst, count=1, flags=re.M)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    text = '/* GENERATED at build time by infamous2-recomp/tools/gen_spu_rc.py (F181) from %s */\n' % \
           os.path.basename(os.path.dirname(os.path.abspath(src_path))) + dst
    old = None
    if os.path.isfile(out_path):
        old = open(out_path, encoding='utf-8', errors='surrogateescape').read()
    if old != text:
        open(out_path, 'w', encoding='utf-8', errors='surrogateescape', newline='\n').write(text)
    print('gen_spu_rc: %s: %d functions cached' % (src_path, nf))

if __name__ == '__main__':
    main()
