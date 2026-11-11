/*
 * ps3recomp - cached getenv() for the RSX hot paths (2026-10-04).
 *
 * The renderer reads its debug switches with getenv() from per-draw and even
 * per-vertex code. On Windows every getenv() takes the CRT environment lock and
 * scans the whole block; with ~10,000 draws per frame (inFamous) that is tens of
 * thousands of locked scans a frame. Every call site passes a string LITERAL, so
 * the literal's address is a stable key: look each one up once and remember it.
 * The environment is not modified at run time. Include AFTER <stdlib.h>.
 * Calls with a computed (non-literal) name must use (getenv)(name) explicitly.
 */
#ifndef PS3RECOMP_RSX_GETENV_CACHE_H
#define PS3RECOMP_RSX_GETENV_CACHE_H
#include <stdlib.h>
#include <stdint.h>

static inline char* rsx_cached_getenv(const char* name)
{
    enum { N = 1024 };
    static const char* volatile s_keys[N];
    static char* s_vals[N];
    uintptr_t h = ((uintptr_t)name >> 2) * (uintptr_t)2654435761u;
    for (unsigned i = 0; i < N; i++) {
        unsigned k = (unsigned)((h + i) & (N - 1));
        const char* kk = s_keys[k];
        if (kk == name) return s_vals[k];
        if (kk == (const char*)1) return (getenv)(name);   /* slot being filled by another thread */
        if (!kk) {
            char* v = (getenv)(name);
            if (__sync_bool_compare_and_swap(&s_keys[k], (const char*)0, (const char*)1)) {
                s_vals[k] = v;
                __sync_synchronize();
                s_keys[k] = name;
            }
            return v;
        }
    }
    return (getenv)(name);
}
#define getenv(n) rsx_cached_getenv(n)

#endif
