#ifndef RSX_PARTICLE_RESOLUTION_H
#define RSX_PARTICLE_RESOLUTION_H
/* Opt-in inFamous particle target scaling. Guest texture coordinates and
 * pitches stay in guest units; only GPU attachments/viewports are enlarged. */
static void rsx_particle_size(unsigned* w, unsigned* h, int enabled,
                              unsigned display_w, unsigned display_h)
{
    if (enabled && *w == 512 && *h == 288 && display_w == 1280 && display_h == 720) {
        *w = display_w; *h = display_h;
    }
}
#endif
