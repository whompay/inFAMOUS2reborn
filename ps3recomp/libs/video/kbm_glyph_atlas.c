#include "kbm_glyph_atlas.h"
#include "cellPad.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct GlyphCell {
    int x0, y0, x1, y1;
    const char* action;
} GlyphCell;

/* Pixel cells in the game's 512x256 UI atlas. Several buttons have both a
 * compact prompt form and a controls-screen form; both point at the same
 * binding. Coordinates deliberately include the transparent tile padding. */
static const GlyphCell s_cells[] = {
    {240,   0, 306,  50, "l1"},       {306,   0, 372,  50, "l2"},
    {372,   0, 438,  50, "r1"},       {438,   0, 504,  50, "r2"},
    {328,  50, 392, 114, "r3"},       {394,  50, 452, 114, "circle"},
    {452,  50, 510, 114, "square"},   {  0, 120,  64, 184, "start"},
    { 64, 120, 128, 184, "select"},   {194, 120, 260, 184, "l3"},
    {  0, 184,  56, 240, "triangle"}, { 56, 184, 112, 240, "cross"},
};

/* Five column, seven row font. Bit zero is the top row. */
static const uint8_t* letter(char c)
{
    static const uint8_t A[5]={0x7e,0x09,0x09,0x09,0x7e};
    static const uint8_t B[5]={0x7f,0x49,0x49,0x49,0x36};
    static const uint8_t C[5]={0x3e,0x41,0x41,0x41,0x22};
    static const uint8_t D[5]={0x7f,0x41,0x41,0x22,0x1c};
    static const uint8_t E[5]={0x7f,0x49,0x49,0x49,0x41};
    static const uint8_t F[5]={0x7f,0x09,0x09,0x09,0x01};
    static const uint8_t G[5]={0x3e,0x41,0x49,0x49,0x7a};
    static const uint8_t H[5]={0x7f,0x08,0x08,0x08,0x7f};
    static const uint8_t I[5]={0x41,0x41,0x7f,0x41,0x41};
    static const uint8_t J[5]={0x20,0x40,0x41,0x3f,0x01};
    static const uint8_t K[5]={0x7f,0x08,0x14,0x22,0x41};
    static const uint8_t L[5]={0x7f,0x40,0x40,0x40,0x40};
    static const uint8_t M[5]={0x7f,0x02,0x0c,0x02,0x7f};
    static const uint8_t N[5]={0x7f,0x02,0x04,0x08,0x7f};
    static const uint8_t O[5]={0x3e,0x41,0x41,0x41,0x3e};
    static const uint8_t P[5]={0x7f,0x09,0x09,0x09,0x06};
    static const uint8_t Q[5]={0x3e,0x41,0x51,0x21,0x5e};
    static const uint8_t R[5]={0x7f,0x09,0x19,0x29,0x46};
    static const uint8_t S[5]={0x46,0x49,0x49,0x49,0x31};
    static const uint8_t T[5]={0x01,0x01,0x7f,0x01,0x01};
    static const uint8_t U[5]={0x3f,0x40,0x40,0x40,0x3f};
    static const uint8_t V[5]={0x1f,0x20,0x40,0x20,0x1f};
    static const uint8_t W[5]={0x7f,0x20,0x18,0x20,0x7f};
    static const uint8_t X[5]={0x63,0x14,0x08,0x14,0x63};
    static const uint8_t Y[5]={0x07,0x08,0x70,0x08,0x07};
    static const uint8_t Z[5]={0x61,0x51,0x49,0x45,0x43};
    static const uint8_t N0[5]={0x3e,0x51,0x49,0x45,0x3e};
    static const uint8_t N1[5]={0x00,0x42,0x7f,0x40,0x00};
    static const uint8_t N2[5]={0x62,0x51,0x49,0x49,0x46};
    static const uint8_t N3[5]={0x22,0x41,0x49,0x49,0x36};
    static const uint8_t N4[5]={0x18,0x14,0x12,0x7f,0x10};
    static const uint8_t N5[5]={0x4f,0x49,0x49,0x49,0x31};
    static const uint8_t N6[5]={0x3e,0x49,0x49,0x49,0x30};
    static const uint8_t N7[5]={0x01,0x71,0x09,0x05,0x03};
    static const uint8_t N8[5]={0x36,0x49,0x49,0x49,0x36};
    static const uint8_t N9[5]={0x06,0x49,0x49,0x49,0x3e};
    switch (c) {
    case 'A': return A; case 'B': return B; case 'C': return C; case 'D': return D;
    case 'E': return E; case 'F': return F; case 'G': return G; case 'H': return H;
    case 'I': return I; case 'J': return J; case 'K': return K; case 'L': return L;
    case 'M': return M; case 'N': return N; case 'O': return O; case 'P': return P;
    case 'Q': return Q; case 'R': return R; case 'S': return S; case 'T': return T;
    case 'U': return U; case 'V': return V; case 'W': return W; case 'X': return X;
    case 'Y': return Y; case 'Z': return Z; case '0': return N0; case '1': return N1;
    case '2': return N2; case '3': return N3; case '4': return N4; case '5': return N5;
    case '6': return N6; case '7': return N7; case '8': return N8; case '9': return N9;
    default: return NULL;
    }
}

static void compact_label(char* s)
{
    if (!strcmp(s, "LMB")) strcpy(s, "M1");
    else if (!strcmp(s, "RMB")) strcpy(s, "M2");
    else if (!strcmp(s, "MMB")) strcpy(s, "M3");
    else if (!strcmp(s, "SPACE")) strcpy(s, "SPC");
    else if (!strcmp(s, "LSHIFT") || !strcmp(s, "RSHIFT") || !strcmp(s, "SHIFT")) strcpy(s, "SHFT");
    else if (!strcmp(s, "LCTRL") || !strcmp(s, "RCTRL") || !strcmp(s, "CTRL")) strcpy(s, "CTRL");
    else if (!strcmp(s, "RETURN") || !strcmp(s, "ENTER")) strcpy(s, "ENT");
    else if (!strcmp(s, "ESCAPE")) strcpy(s, "ESC");
    else if (!strcmp(s, "BACKSPACE")) strcpy(s, "BKSP");
    else if (!strcmp(s, "CAPSLOCK")) strcpy(s, "CAPS");
    else if (!strcmp(s, "WHEELUP")) strcpy(s, "WU");
    else if (!strcmp(s, "WHEELDOWN")) strcpy(s, "WD");
}

static uint8_t* pixel(uint8_t* rgba, uint32_t pitch, int x, int y)
{
    return rgba + (size_t)y * pitch + (size_t)x * 4u;
}

static void set_alpha(uint8_t* rgba, uint32_t pitch, int x, int y, uint8_t a)
{
    uint8_t* p = pixel(rgba, pitch, x, y);
    p[0] = p[1] = p[2] = 255; p[3] = a;
}

static void clear_cell(uint8_t* rgba, uint32_t pitch, uint32_t w, uint32_t h,
                       const GlyphCell* c, int* x0, int* y0, int* x1, int* y1)
{
    *x0 = c->x0 * (int)w / 512; *x1 = c->x1 * (int)w / 512;
    *y0 = c->y0 * (int)h / 256; *y1 = c->y1 * (int)h / 256;
    if (*x1 <= *x0 || *y1 <= *y0) return;
    for (int y = *y0; y < *y1; y++)
        for (int x = *x0; x < *x1; x++) set_alpha(rgba, pitch, x, y, 0);
}

static int inside_round_rect(int x, int y, int x0, int y0, int x1, int y1, int radius)
{
    if (x < x0 || x >= x1 || y < y0 || y >= y1) return 0;
    int dx = x < x0 + radius ? x0 + radius - x : x >= x1 - radius ? x - (x1 - radius - 1) : 0;
    int dy = y < y0 + radius ? y0 + radius - y : y >= y1 - radius ? y - (y1 - radius - 1) : 0;
    return !dx || !dy || dx * dx + dy * dy <= radius * radius;
}

static int mouse_button(const char* label)
{
    if (!strcmp(label, "LMB") || !strcmp(label, "MOUSE1")) return 1;
    if (!strcmp(label, "RMB") || !strcmp(label, "MOUSE2")) return 2;
    if (!strcmp(label, "MMB") || !strcmp(label, "MOUSE3")) return 3;
    if (!strcmp(label, "MB4") || !strcmp(label, "MOUSE4")) return 4;
    if (!strcmp(label, "MB5") || !strcmp(label, "MOUSE5")) return 5;
    return 0;
}

/* A compact mouse silhouette. The bound button is solid while the shell and
 * button divisions remain visible, which reads much more naturally in a
 * prompt than M1/M2/M3 text. */
static void draw_mouse(uint8_t* rgba, uint32_t pitch, uint32_t w, uint32_t h,
                       const GlyphCell* c, int selected)
{
    int x0, y0, x1, y1;
    clear_cell(rgba, pitch, w, h, c, &x0, &y0, &x1, &y1);
    if (x1 <= x0 || y1 <= y0) return;

    const int cw = x1 - x0, ch = y1 - y0;
    if (cw < 4 || ch < 4) {
        set_alpha(rgba, pitch, (x0 + x1) / 2, (y0 + y1) / 2, 255);
        return;
    }
    int mw = cw * 9 / 20, mh = ch * 4 / 5;
    if (mw < 3) mw = 3; if (mh < 4) mh = 4;
    int mx0 = (x0 + x1 - mw) / 2, mx1 = mx0 + mw;
    int my0 = (y0 + y1 - mh) / 2, my1 = my0 + mh;
    int radius = mw / 3, thick = cw / 22;
    if (thick < 1) thick = 1;
    int split = my0 + mh * 9 / 20, mid = (mx0 + mx1) / 2;

    for (int y = my0; y < my1; y++) for (int x = mx0; x < mx1; x++) {
        int outer = inside_round_rect(x, y, mx0, my0, mx1, my1, radius);
        int inner = inside_round_rect(x, y, mx0 + thick, my0 + thick,
                                      mx1 - thick, my1 - thick,
                                      radius > thick ? radius - thick : 1);
        int fill = inner && y < split &&
                   ((selected == 1 && x < mid) || (selected == 2 && x > mid));
        if (outer && (!inner || fill)) set_alpha(rgba, pitch, x, y, 255);
    }

    /* Transparent seams keep the selected half legible as a distinct button. */
    for (int y = my0 + thick; y < split; y++)
        for (int x = mid - thick / 2; x <= mid + thick / 2; x++)
            set_alpha(rgba, pitch, x, y, 0);
    for (int y = split - thick / 2; y <= split + thick / 2; y++)
        for (int x = mx0 + thick; x < mx1 - thick; x++)
            set_alpha(rgba, pitch, x, y, 255);

    int ww = mw / 5, wh = mh / 5;
    if (ww < 1) ww = 1; if (wh < 1) wh = 1;
    int wx0 = mid - ww / 2, wy0 = my0 + mh / 7;
    for (int y = wy0; y < wy0 + wh; y++)
        for (int x = wx0; x < wx0 + ww; x++)
            if (inside_round_rect(x, y, wx0, wy0, wx0 + ww, wy0 + wh, ww / 2))
                set_alpha(rgba, pitch, x, y, selected == 3 ? 255 : 0);

    if (selected == 4 || selected == 5) {
        int sy = my0 + mh * (selected == 4 ? 11 : 14) / 20;
        int sx0 = mx0 - thick, sx1 = mx0 + thick * 2;
        for (int y = sy; y < sy + thick * 2; y++)
            for (int x = sx0; x < sx1; x++) set_alpha(rgba, pitch, x, y, 255);
    }
}

static void draw_keycap(uint8_t* rgba, uint32_t pitch, uint32_t w, uint32_t h,
                        const GlyphCell* c, const char* label_text)
{
    int x0, y0, x1, y1;
    clear_cell(rgba, pitch, w, h, c, &x0, &y0, &x1, &y1);
    if (x1 <= x0 || y1 <= y0) return;

    int mx = (x1 - x0) / 10, my = (y1 - y0) / 6;
    if (mx < 1) mx = 1; if (my < 1) my = 1;
    int bx0 = x0 + mx, bx1 = x1 - mx, by0 = y0 + my, by1 = y1 - my;
    int radius = (by1 - by0) / 5;
    for (int y = by0; y < by1; y++) for (int x = bx0; x < bx1; x++) {
        int dx = x < bx0 + radius ? bx0 + radius - x : x >= bx1 - radius ? x - (bx1 - radius - 1) : 0;
        int dy = y < by0 + radius ? by0 + radius - y : y >= by1 - radius ? y - (by1 - radius - 1) : 0;
        if (!dx || !dy || dx * dx + dy * dy <= radius * radius)
            set_alpha(rgba, pitch, x, y, 255);
    }

    char label[24]; strncpy(label, label_text, sizeof label - 1); label[sizeof label - 1] = 0;
    compact_label(label);
    const int n = (int)strlen(label);
    if (!n) return;
    int scale_x = (bx1 - bx0 - 6) / (n * 6 - 1);
    int scale_y = (by1 - by0 - 6) / 7;
    int scale = scale_x < scale_y ? scale_x : scale_y;
    if (scale < 1) scale = 1; if (scale > 4) scale = 4;
    const int tw = (n * 6 - 1) * scale, th = 7 * scale;
    const int tx = (bx0 + bx1 - tw) / 2, ty = (by0 + by1 - th) / 2;
    for (int i = 0; i < n; i++) {
        const uint8_t* bits = letter(label[i]);
        if (!bits) continue;
        for (int col = 0; col < 5; col++) for (int row = 0; row < 7; row++)
            if (bits[col] & (1u << row))
                for (int yy = 0; yy < scale; yy++) for (int xx = 0; xx < scale; xx++) {
                    int px = tx + (i * 6 + col) * scale + xx, py = ty + row * scale + yy;
                    if (px >= bx0 && px < bx1 && py >= by0 && py < by1)
                        set_alpha(rgba, pitch, px, py, 0);
                }
    }
}

static int is_controller_atlas(const uint8_t* rgba, uint32_t pitch, uint32_t w, uint32_t h)
{
    if (w != 512 || h != 256) return 0;
    /* Both the controller sheet and the ordinary Unicode font sheet are
     * 512x256 RGBA atlases. A single broad coverage test therefore also
     * matched the font sheet and replaced letters with keycaps. Match the
     * controller sheet's three independent icon groups instead: shoulder
     * labels across the top, face buttons at lower-left, and the large
     * square/lightning/help group at lower-right. */
    unsigned shoulders = 0, faces = 0, utility = 0;
    for (int y = 0; y < 50; y += 2)
        for (int x = 240; x < 504; x += 2)
            if (rgba[(size_t)y * pitch + (size_t)x * 4u + 3] >= 64) shoulders++;
    for (int y = 184; y < 240; y += 2)
        for (int x = 0; x < 112; x += 2)
            if (rgba[(size_t)y * pitch + (size_t)x * 4u + 3] >= 64) faces++;
    for (int y = 118; y < 224; y += 2)
        for (int x = 400; x < 502; x += 2)
            if (rgba[(size_t)y * pitch + (size_t)x * 4u + 3] >= 64) utility++;
    return shoulders > 1000 && faces > 350 && utility > 500;
}

int kbm_glyph_atlas_patch(uint8_t* rgba, uint32_t pitch,
                          uint32_t width, uint32_t height, int force)
{
    if (!rgba || !pitch) return 0;
    if (!force && !is_controller_atlas(rgba, pitch, width, height)) return 0;
    static int enabled = -1;
    if (enabled < 0) { const char* e = getenv("KBM_GLYPHS"); enabled = !(e && *e == '0'); }
    const int device = cellPadHostInputDevice();
    if (!force) {
        static int last_reported_device = -1;
        if (device != last_reported_device) {
            fprintf(stderr, "[KBM glyphs] native prompt atlas: %s glyphs\n",
                    device == CELL_PAD_HOST_DEVICE_KBM ? "keyboard/mouse" : "controller");
            last_reported_device = device;
        }
    }
    if (!enabled || device != CELL_PAD_HOST_DEVICE_KBM) return 1;
    for (unsigned i = 0; i < sizeof s_cells / sizeof s_cells[0]; i++) {
        char label[24];
        if (cellPadGetKbmBindingLabel(s_cells[i].action, label, sizeof label)) {
            int mb = mouse_button(label);
            if (mb) draw_mouse(rgba, pitch, width, height, &s_cells[i], mb);
            else    draw_keycap(rgba, pitch, width, height, &s_cells[i], label);
        }
    }
    return 1;
}

