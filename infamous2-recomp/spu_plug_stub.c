#include <stdint.h>
#include "spu_context.h"

typedef void (*spu_plug_fn)(spu_context*);

/* Replaced automatically by spu/wwsplug/spu_recomp.c once the title's
 * relocatable WWS job plugins have been identified and lifted. */
spu_plug_fn spu_plug_lookup(spu_context* ctx, uint32_t pc)
{
    (void)ctx;
    (void)pc;
    return 0;
}
