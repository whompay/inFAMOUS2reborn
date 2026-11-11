#pragma once
#include <stdint.h>
/* Configure once before guest threads start. Reports attempted 32-bit stores
 * before committing them. Observer must not modify guest memory/call guest code.
 * A zero size disables it without disabling the VM's fast path. */
#ifdef __cplusplus
extern "C" {
#endif
typedef struct PpuStoreObserver {
    uint32_t begin, size;
    void (*notify)(uint32_t address, uint32_t value, uint32_t caller_lr);
} PpuStoreObserver;
extern PpuStoreObserver g_ppu_store_observer;
#ifdef __cplusplus
}
#endif
