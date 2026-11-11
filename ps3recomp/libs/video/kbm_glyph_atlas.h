#ifndef PS3RECOMP_KBM_GLYPH_ATLAS_H
#define PS3RECOMP_KBM_GLYPH_ATLAS_H

#include <stdint.h>

/* Detects inFamous 2's controller atlas and, in KBM mode, replaces its button
 * cells with keycaps. `force` is used for mip levels after level zero matched. */
int kbm_glyph_atlas_patch(uint8_t* rgba, uint32_t pitch,
                          uint32_t width, uint32_t height, int force);

#endif

