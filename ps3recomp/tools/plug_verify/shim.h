#ifndef _WIN32
#include <stdint.h>
static inline long InterlockedExchangeAdd(volatile long* p, long v) { return __sync_fetch_and_add(p, v); }
static inline long InterlockedIncrement(volatile long* p) { return __sync_add_and_fetch(p, 1); }
static inline long InterlockedExchange(volatile long* p, long v) { return __sync_lock_test_and_set(p, v); }
static inline long long InterlockedExchange64(volatile long long* p, long long v) { return __sync_lock_test_and_set(p, v); }
static inline unsigned long long GetTickCount64(void) { return 0; }
#endif
