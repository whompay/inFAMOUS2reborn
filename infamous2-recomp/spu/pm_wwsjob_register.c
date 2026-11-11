#include <stdint.h>
#include "spu_context.h"

extern void spu_begin_image(int image_id);
extern void spu_workload_register_img(uint64_t fingerprint,
                                      void (*entry)(spu_context*),
                                      int image_id,
                                      const char* name);
extern void pm_wwsjob_spu_recomp_register(void);
extern void pm_wwsjob_spu_func_00002B60(spu_context*);

static void infamous2_register_wws_policy(void)
{
    spu_begin_image(100);
    pm_wwsjob_spu_recomp_register();
    spu_workload_register_img(0xB3523C4B17650EF8ULL,
                              pm_wwsjob_spu_func_00002B60,
                              100,
                              "inFamous2 pm_wwsjob");
    spu_begin_image(0);
}

#if defined(_MSC_VER)
#pragma section(".CRT$XCU", read)
static void __cdecl infamous2_register_wws_policy_ctor(void)
{
    infamous2_register_wws_policy();
}
__declspec(allocate(".CRT$XCU")) void (__cdecl* infamous2_register_wws_policy_ctor_ptr)(void) =
    infamous2_register_wws_policy_ctor;
#else
__attribute__((constructor)) static void infamous2_register_wws_policy_ctor(void)
{
    infamous2_register_wws_policy();
}
#endif
