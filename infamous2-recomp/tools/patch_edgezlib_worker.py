"""Apply the inFamous 2 EDGE-zlib task continuation to a generated SPU lift.

The retail worker asserts at edgezlib_inflate_task.cpp:272 when the SDK queue
completion helper reports 0x80410914.  The lifted/HLE queue can expose this
transient once during the load-menu handoff.  Continuing at the instruction
after the assertion keeps the persistent worker alive and matches the working
timing observed in native runs.
"""

from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
marker = "[infamous-edgezlib] queue completion"
if marker in source:
    raise SystemExit(0)

include_anchor = '#include "spu_helpers.h"\n'
function_anchor = "void spu_0000_at_0079B980_spu_func_00003324(spu_context* ctx) {\n"
if source.count(include_anchor) != 1 or source.count(function_anchor) != 1:
    raise SystemExit("EDGE-zlib patch anchors did not match exactly once")

source = source.replace(
    include_anchor,
    include_anchor + "#include <stdlib.h>\n#include <stdio.h>\n",
)
source = source.replace(
    function_anchor,
    function_anchor
    + "        /* Title-scoped recovery for the one transient queue completion error. */\n"
    + "        { static unsigned _edge_qerr_n = 0;\n"
    + "          if (_edge_qerr_n++ < 16)\n"
    + "              fprintf(stderr, \"[infamous-edgezlib] queue completion rc=0x%08X at edgezlib_inflate_task.cpp:272\\n\",\n"
    + "                      ctx->gpr[3]._u32[0]); }\n"
    + "        { const char* _edge_continue = getenv(\"INFAMOUS_EDGEZLIB_CONTINUE\");\n"
    + "          if (_edge_continue && *_edge_continue != '0') {\n"
    + "            ctx->pc = 0x31C8;\n"
    + "            g_spu_trampoline_fn = spu_0000_at_0079B980_spu_func_000031C8;\n"
    + "            return;\n"
    + "          } }\n",
)
path.write_text(source, encoding="utf-8", newline="\n")

