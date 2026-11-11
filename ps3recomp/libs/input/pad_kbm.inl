/*
 * pad_kbm.inl -- keyboard + mouse ("KBM") -> the emulated DualShock 3 on pad port 0.
 * #included by cellPad.c (Win32/XInput build only).
 *
 * The guest only ever sees cellPad. This layer turns host keys, mouse buttons,
 * the mouse wheel and RAW mouse motion into pad port 0:
 *   - WASD -> left stick (full deflection; hold `walk` for partial),
 *   - raw mouse deltas (WM_INPUT, RegisterRawInputDevices on a message-only
 *     window owned by a dedicated KBM thread) -> right stick, as a VELOCITY:
 *     counts since the previous delivered pad packet / elapsed time, times a
 *     sensitivity, optional acceleration/curve/smoothing, radial deadzone
 *     compensation, invert-Y. The guest turns its camera by stick x frame time,
 *     so "displacement per read / time per read" maps mouse travel to turn angle,
 *     whatever the frame rate.
 *   - keys / mouse buttons / wheel -> face buttons, shoulders, triggers (digital
 *     bit + analog pressure 255), d-pad, L3/R3, START/SELECT.
 *
 * Threads and cost. The KBM thread wakes every KBM_TICK_MS, drains its raw
 * input, reads the bound keys with GetAsyncKeyState (only while the game
 * window is the foreground window) and publishes one 64-bit snapshot
 * (InterlockedExchange64). The pad poll (pad thread / GetData) only reads that
 * snapshot; cellPadGetData only does two InterlockedExchange on the mouse
 * counters for a delivered packet. No locks, no getenv after init, no
 * per-poll logging (diagnostics are throttled [KBM] lines, `log=0` silences).
 *
 * Cursor: while the game window is focused and mouse-look is on, the cursor
 * is hidden (WM_SETCURSOR in a runtime subclass of the backend window -- no
 * backend source edit) and clipped to the window centre. Released on focus
 * loss / Alt-Tab / minimise, and by `toggle_capture` (F9); a left click in
 * the window recaptures. Raw mouse input is registered only while captured.
 *
 * Esc: the backend closes the game on Esc. When Esc is bound (default: START)
 * and KBM is focused, Esc is swallowed; Shift+Esc still quits.
 *
 * Off switches: KBM=0 (old behaviour exactly: legacy keyboard fallback, Esc
 * quits). KBM is also OFF by default whenever PAD_SCRIPT / PAD_FILE /
 * PAD_AUTOPRESS / PAD_NO_KEYBOARD is set (the test harness), so scripted runs
 * never grab the user's cursor; KBM=1 forces it on.
 *
 * Config: kbm.ini (KBM_INI=<path>, else ./kbm.ini, <exe dir>/kbm.ini,
 * <exe dir>/../kbm.ini), hot-reloaded within ~1 s of a save. Every [kbm]
 * option has an env override KBM_<OPTION> (e.g. KBM_MOUSE_SENS=3), every
 * binding KBM_BIND_<ACTION> (e.g. KBM_BIND_CROSS=Space,J).
 *
 * Test: KBM_TEST="<t0>-<t1>:<item>;..." (seconds since cellPadInit) injects
 * synthetic input INTO THIS PIPELINE (same mapping, same mouse->stick
 * conversion) without touching the real devices: <item> = an action name
 * (cross, l1, ls_up, ...), mouse_x=<counts/s>, mouse_y=<counts/s>,
 * wheel_up, wheel_down. In test mode real input is ignored and the cursor is
 * never captured, unless KBM_TEST_REAL=1.
 */

#include <stddef.h>

#define KBM_TICK_MS     4
#define KBM_MAX_KEYS    4
#define KBM_MAX_TEST    32
#define KVK_WHEEL_UP    0x0F01
#define KVK_WHEEL_DOWN  0x0F02

enum {
    /* 0..15 = CELL_PAD_CTRL_* bit index (select .. square) */
    KA_LS_UP = 16, KA_LS_DOWN, KA_LS_LEFT, KA_LS_RIGHT,
    KA_RS_UP, KA_RS_DOWN, KA_RS_LEFT, KA_RS_RIGHT,
    KA_RUN, KA_TOGGLE, KA_COUNT
};
static const char* const s_kbm_act_name[KA_COUNT] = {
    "select", "l3", "r3", "start", "up", "right", "down", "left",
    "l2", "r2", "l1", "r1", "triangle", "circle", "cross", "square",
    "ls_up", "ls_down", "ls_left", "ls_right",
    "rs_up", "rs_down", "rs_left", "rs_right",
    "run", "toggle_capture"
};

typedef struct {
    int   enabled, capture, invert_y, log, wheel_ms, esc_bound;
    float sens, sens_y, curve, accel, deadzone, smooth, walk_scale, look_keys_scale, carry;
    unsigned short key[KA_COUNT][KBM_MAX_KEYS];
} KbmCfg;

/* Settings (kbm.ini [kbm] section; env KBM_<NAME upper-cased>). */
static const struct { const char* n; int is_float; size_t off; } s_kbm_opt[] = {
    { "enabled",         0, offsetof(KbmCfg, enabled) },
    { "capture",         0, offsetof(KbmCfg, capture) },
    { "invert_y",        0, offsetof(KbmCfg, invert_y) },
    { "log",             0, offsetof(KbmCfg, log) },
    { "wheel_ms",        0, offsetof(KbmCfg, wheel_ms) },
    { "mouse_sens",      1, offsetof(KbmCfg, sens) },
    { "mouse_sens_y",    1, offsetof(KbmCfg, sens_y) },
    { "mouse_curve",     1, offsetof(KbmCfg, curve) },
    { "mouse_accel",     1, offsetof(KbmCfg, accel) },
    { "mouse_deadzone",  1, offsetof(KbmCfg, deadzone) },
    { "mouse_smooth",    1, offsetof(KbmCfg, smooth) },
    { "mouse_carry",     1, offsetof(KbmCfg, carry) },
    { "walk_scale",      1, offsetof(KbmCfg, walk_scale) },
    { "look_keys_scale", 1, offsetof(KbmCfg, look_keys_scale) },
};

/* Built-in defaults, in kbm.ini syntax (a kbm.ini only needs what it changes).
 * Layout follows the inFamous 2 manual: X jump/climb (aiming: blast), Square
 * melee (aiming: grenade), Triangle interact/finisher (aiming: rocket), Circle
 * drop/cover/dive roll, L1 aim, R1 lock-on (aiming: bolt), L2 electric drain,
 * R2 kinetic pulse, R3 switch reticle side, L3 radar pulse, SELECT map,
 * START pause, d-pad up precision. */
static const char s_kbm_default_ini[] =
    "[kbm]\n"
    "enabled=1\n" "capture=1\n" "invert_y=0\n" "log=1\n" "wheel_ms=80\n"
    "mouse_sens=6\n" "mouse_sens_y=0.8\n" "mouse_curve=0.8\n" "mouse_accel=0\n"
    "mouse_deadzone=0.12\n" "mouse_smooth=0\n" "mouse_carry=0.4\n" "walk_scale=0.45\n" "look_keys_scale=0.7\n"
    "[bind]\n"
    "ls_up=W\n" "ls_down=S\n" "ls_left=A\n" "ls_right=D\n" "run=LShift\n"
    "cross=Space\n" "circle=C,Backspace\n" "square=E\n" "triangle=F\n"
    "l1=RMB\n" "r1=LMB\n" "l2=Q\n" "r2=R\n" "l3=V\n" "r3=MMB,X\n"
    "up=1,Up,MB5,WheelUp\n" "right=2,Right\n" "down=3,Down,MB4,WheelDown\n" "left=4,Left\n"
    "start=Esc,Enter\n" "select=Tab,M\n"
    "rs_up=none\n" "rs_down=none\n" "rs_left=none\n" "rs_right=none\n"
    "toggle_capture=F9\n";

static const struct { const char* n; unsigned short vk; } s_kbm_keyname[] = {
    { "LMB", VK_LBUTTON }, { "MOUSE1", VK_LBUTTON }, { "RMB", VK_RBUTTON }, { "MOUSE2", VK_RBUTTON },
    { "MMB", VK_MBUTTON }, { "MOUSE3", VK_MBUTTON }, { "MB4", VK_XBUTTON1 }, { "MOUSE4", VK_XBUTTON1 },
    { "MB5", VK_XBUTTON2 }, { "MOUSE5", VK_XBUTTON2 },
    { "WHEELUP", KVK_WHEEL_UP }, { "WHEELDOWN", KVK_WHEEL_DOWN },
    { "SPACE", VK_SPACE }, { "ENTER", VK_RETURN }, { "RETURN", VK_RETURN }, { "ESC", VK_ESCAPE },
    { "ESCAPE", VK_ESCAPE }, { "TAB", VK_TAB }, { "BACKSPACE", VK_BACK }, { "CAPSLOCK", VK_CAPITAL },
    { "LSHIFT", VK_LSHIFT }, { "RSHIFT", VK_RSHIFT }, { "SHIFT", VK_SHIFT },
    { "LCTRL", VK_LCONTROL }, { "RCTRL", VK_RCONTROL }, { "CTRL", VK_CONTROL },
    { "LALT", VK_LMENU }, { "RALT", VK_RMENU }, { "ALT", VK_MENU },
    { "UP", VK_UP }, { "DOWN", VK_DOWN }, { "LEFT", VK_LEFT }, { "RIGHT", VK_RIGHT },
    { "INSERT", VK_INSERT }, { "DELETE", VK_DELETE }, { "HOME", VK_HOME }, { "END", VK_END },
    { "PGUP", VK_PRIOR }, { "PGDN", VK_NEXT },
    { "GRAVE", VK_OEM_3 }, { "MINUS", VK_OEM_MINUS }, { "EQUALS", VK_OEM_PLUS },
    { "LBRACKET", VK_OEM_4 }, { "RBRACKET", VK_OEM_6 }, { "BACKSLASH", VK_OEM_5 },
    { "SEMICOLON", VK_OEM_1 }, { "QUOTE", VK_OEM_7 }, { "COMMA", VK_OEM_COMMA },
    { "PERIOD", VK_OEM_PERIOD }, { "SLASH", VK_OEM_2 },
    { "NUMPLUS", VK_ADD }, { "NUMMINUS", VK_SUBTRACT }, { "NUMSTAR", VK_MULTIPLY },
    { "NUMSLASH", VK_DIVIDE }, { "NUMDOT", VK_DECIMAL },
};

static KbmCfg          s_kbm_cfgbuf[2];
static KbmCfg* volatile s_kbm_cfg = &s_kbm_cfgbuf[0];
static int             s_kbm_on = 0;          /* KBM thread running; set once in kbm_init */
static int             s_kbm_test = 0, s_kbm_test_real = 0;
static volatile LONG   s_kbm_mx = 0, s_kbm_my = 0;   /* raw counts not yet turned into stick */
static volatile LONG64 s_kbm_pub = 0;                 /* snapshot, see kbm_publish */
static volatile LONG   s_kbm_focused = 0, s_kbm_captured = 0, s_kbm_mlook = 0;
static volatile LONG   s_kbm_run = 0;
static HANDLE          s_kbm_thread = NULL;
static HWND            s_kbm_game = NULL, s_kbm_sub_hwnd = NULL, s_kbm_msgwnd = NULL;
static WNDPROC         s_kbm_prev_proc = NULL;
static char            s_kbm_ini[MAX_PATH];
static FILETIME        s_kbm_ini_mtime;
static int             s_kbm_ini_exists = 0;
static LARGE_INTEGER   s_kbm_qpf, s_kbm_t0;

static double kbm_now_s(void)
{
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)(t.QuadPart - s_kbm_t0.QuadPart) / (double)s_kbm_qpf.QuadPart;
}

/* ---- config parsing ------------------------------------------------------ */

static void kbm_trim(char** s, char** e)
{
    while (*s < *e && (**s == ' ' || **s == '\t' || **s == '\r' || **s == '"')) (*s)++;
    while (*e > *s && ((*e)[-1] == ' ' || (*e)[-1] == '\t' || (*e)[-1] == '\r' || (*e)[-1] == '"')) (*e)--;
}

static unsigned short kbm_parse_key(const char* s, size_t n)
{
    char u[24];
    if (n == 0 || n >= sizeof u) return 0;
    for (size_t i = 0; i < n; i++) u[i] = (char)((s[i] >= 'a' && s[i] <= 'z') ? s[i] - 32 : s[i]);
    u[n] = 0;
    if (n == 1 && ((u[0] >= 'A' && u[0] <= 'Z') || (u[0] >= '0' && u[0] <= '9'))) return (unsigned short)u[0];
    if (u[0] == '0' && u[1] == 'X') { unsigned long v = strtoul(u, NULL, 16); return (unsigned short)(v > 0 && v < 0xFF ? v : 0); }
    if (u[0] == 'F' && n <= 3 && u[1] >= '1' && u[1] <= '9') {
        int f = atoi(u + 1); if (f >= 1 && f <= 24) return (unsigned short)(VK_F1 + f - 1);
    }
    if (!strncmp(u, "NUM", 3) && n == 4 && u[3] >= '0' && u[3] <= '9') return (unsigned short)(VK_NUMPAD0 + (u[3] - '0'));
    for (unsigned i = 0; i < sizeof s_kbm_keyname / sizeof s_kbm_keyname[0]; i++)
        if (!strcmp(u, s_kbm_keyname[i].n)) return s_kbm_keyname[i].vk;
    return 0;
}

static int kbm_find_action(const char* s, size_t n)
{
    /* `walk` was the original name for the speed modifier. Continue accepting
     * it in old configs, but its behavior is now the inverse: normal movement
     * walks and the modifier runs. */
    if (n == 4 && !_strnicmp(s, "walk", 4)) return KA_RUN;
    for (int a = 0; a < KA_COUNT; a++)
        if (strlen(s_kbm_act_name[a]) == n && !_strnicmp(s, s_kbm_act_name[a], n)) return a;
    return -1;
}

/* value = "Key[,Key...]" or "none"; replaces the action's list. */
static void kbm_set_binding(KbmCfg* c, int a, const char* v, const char* src)
{
    memset(c->key[a], 0, sizeof c->key[a]);
    int k = 0;
    const char* p = v;
    while (*p) {
        const char* q = p; while (*q && *q != ',') q++;
        char* s = (char*)p; char* e = (char*)q; kbm_trim(&s, &e);
        if (e > s && !(e - s == 4 && !_strnicmp(s, "none", 4))) {
            unsigned short vk = kbm_parse_key(s, (size_t)(e - s));
            if (!vk) fprintf(stderr, "[KBM] %s: unknown key '%.*s' for %s (ignored)\n", src, (int)(e - s), s, s_kbm_act_name[a]);
            else if (k < KBM_MAX_KEYS) c->key[a][k++] = vk;
            else fprintf(stderr, "[KBM] %s: %s has more than %d keys (extra ignored)\n", src, s_kbm_act_name[a], KBM_MAX_KEYS);
        }
        p = *q ? q + 1 : q;
    }
}

static int kbm_set_option(KbmCfg* c, const char* name, size_t n, const char* v)
{
    for (unsigned i = 0; i < sizeof s_kbm_opt / sizeof s_kbm_opt[0]; i++) {
        if (strlen(s_kbm_opt[i].n) == n && !_strnicmp(name, s_kbm_opt[i].n, n)) {
            char* base = (char*)c + s_kbm_opt[i].off;
            if (s_kbm_opt[i].is_float) *(float*)base = (float)atof(v);
            else                        *(int*)base = atoi(v);
            return 1;
        }
    }
    return 0;
}

static void kbm_parse_text(KbmCfg* c, const char* text, const char* src)
{
    int sect = 0;                          /* 0 = [kbm], 1 = [bind], 2 = other */
    const char* p = text;
    while (*p) {
        const char* eol = p; while (*eol && *eol != '\n') eol++;
        const char* cut = p; while (cut < eol && *cut != ';' && *cut != '#') cut++;
        char* s = (char*)p; char* e = (char*)cut; kbm_trim(&s, &e);
        if (e > s) {
            if (*s == '[') {
                if (e - s >= 6 && !_strnicmp(s, "[bind]", 6)) sect = 1;
                else if (e - s >= 5 && !_strnicmp(s, "[kbm]", 5)) sect = 0;
                else sect = 2;
            } else if (sect != 2) {
                char* eq = s; while (eq < e && *eq != '=') eq++;
                if (eq < e) {
                    char* ks = s; char* ke = eq; kbm_trim(&ks, &ke);
                    char* vs = eq + 1; char* ve = e; kbm_trim(&vs, &ve);
                    char val[256]; size_t vn = (size_t)(ve - vs); if (vn >= sizeof val) vn = sizeof val - 1;
                    memcpy(val, vs, vn); val[vn] = 0;
                    int a = kbm_find_action(ks, (size_t)(ke - ks));
                    if (sect == 1 && a >= 0)             kbm_set_binding(c, a, val, src);
                    else if (sect == 0 && kbm_set_option(c, ks, (size_t)(ke - ks), val)) {}
                    else if (sect == 0 && a >= 0)        kbm_set_binding(c, a, val, src);   /* tolerate */
                    else fprintf(stderr, "[KBM] %s: unknown %s '%.*s' (ignored)\n", src,
                                 sect ? "action" : "option", (int)(ke - ks), ks);
                }
            }
        }
        p = *eol ? eol + 1 : eol;
    }
}

static void kbm_env_overrides(KbmCfg* c)
{
    char name[64];
    for (unsigned i = 0; i < sizeof s_kbm_opt / sizeof s_kbm_opt[0]; i++) {
        snprintf(name, sizeof name, "KBM_%s", s_kbm_opt[i].n);
        for (char* q = name; *q; q++) if (*q >= 'a' && *q <= 'z') *q -= 32;
        const char* v = getenv(name);
        if (v && *v) kbm_set_option(c, s_kbm_opt[i].n, strlen(s_kbm_opt[i].n), v);
    }
    for (int a = 0; a < KA_COUNT; a++) {
        snprintf(name, sizeof name, "KBM_BIND_%s", s_kbm_act_name[a]);
        for (char* q = name; *q; q++) if (*q >= 'a' && *q <= 'z') *q -= 32;
        const char* v = getenv(name);
        if (v) kbm_set_binding(c, a, v, "env");
    }
}

static int kbm_ini_stat(FILETIME* ft)
{
    WIN32_FILE_ATTRIBUTE_DATA fa;
    if (!s_kbm_ini[0] || !GetFileAttributesExA(s_kbm_ini, GetFileExInfoStandard, &fa)) return 0;
    *ft = fa.ftLastWriteTime;
    return 1;
}

static void kbm_load_cfg(int first)
{
    KbmCfg* c = (s_kbm_cfg == &s_kbm_cfgbuf[0]) ? &s_kbm_cfgbuf[1] : &s_kbm_cfgbuf[0];
    memset(c, 0, sizeof *c);
    kbm_parse_text(c, s_kbm_default_ini, "defaults");
    FILETIME ft; memset(&ft, 0, sizeof ft);
    s_kbm_ini_exists = kbm_ini_stat(&ft);
    s_kbm_ini_mtime = ft;
    if (s_kbm_ini_exists) {
        FILE* f = fopen(s_kbm_ini, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
            if (n > 0 && n < (1 << 20)) {
                char* buf = (char*)malloc((size_t)n + 1);
                if (buf) { size_t got = fread(buf, 1, (size_t)n, f); buf[got] = 0;
                           kbm_parse_text(c, buf, s_kbm_ini); free(buf); }
            }
            fclose(f);
        }
    }
    kbm_env_overrides(c);
    if (c->sens_y == 0.0f) c->sens_y = 1.0f;
    if (c->curve <= 0.0f) c->curve = 1.0f;
    if (c->deadzone < 0.0f) c->deadzone = 0.0f;
    if (c->deadzone > 0.9f) c->deadzone = 0.9f;
    if (c->smooth < 0.0f) c->smooth = 0.0f;
    if (c->smooth > 0.95f) c->smooth = 0.95f;
    if (c->wheel_ms < 8) c->wheel_ms = 8;
    c->esc_bound = 0;
    for (int a = 0; a < KA_COUNT; a++)
        for (int k = 0; k < KBM_MAX_KEYS; k++) if (c->key[a][k] == VK_ESCAPE) c->esc_bound = 1;
    MemoryBarrier();
    s_kbm_cfg = c;
    fprintf(stderr, "[KBM] config %s%s: enabled=%d capture=%d sens=%.2f sens_y=%.2f curve=%.2f accel=%.2f "
            "deadzone=%.2f smooth=%.2f invert_y=%d walk=%.2f\n",
            first ? "" : "RELOADED ", s_kbm_ini_exists ? s_kbm_ini : "(built-in defaults; no kbm.ini)",
            c->enabled, c->capture, c->sens, c->sens_y, c->curve, c->accel, c->deadzone, c->smooth,
            c->invert_y, c->walk_scale);
    fflush(stderr);
}

static void kbm_resolve_ini(void)
{
    const char* e = getenv("KBM_INI");
    if (e && *e) { snprintf(s_kbm_ini, sizeof s_kbm_ini, "%s", e); return; }
    char cand[3][MAX_PATH]; int nc = 0;
    snprintf(cand[nc++], MAX_PATH, "kbm.ini");
    char exe[MAX_PATH]; DWORD n = GetModuleFileNameA(NULL, exe, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        char* sl = strrchr(exe, '\\'); if (sl) *sl = 0;
        snprintf(cand[nc++], MAX_PATH, "%s\\kbm.ini", exe);
        snprintf(cand[nc++], MAX_PATH, "%s\\..\\kbm.ini", exe);
    }
    for (int i = 0; i < nc; i++)
        if (GetFileAttributesA(cand[i]) != INVALID_FILE_ATTRIBUTES) {
            if (!GetFullPathNameA(cand[i], MAX_PATH, s_kbm_ini, NULL)) snprintf(s_kbm_ini, MAX_PATH, "%s", cand[i]);
            return;
        }
    /* None yet: watch ./kbm.ini so one created later is picked up. */
    if (!GetFullPathNameA("kbm.ini", MAX_PATH, s_kbm_ini, NULL)) snprintf(s_kbm_ini, MAX_PATH, "kbm.ini");
}

static void kbm_cfg_check_reload(void)
{
    FILETIME ft; memset(&ft, 0, sizeof ft);
    int ex = kbm_ini_stat(&ft);
    if (ex != s_kbm_ini_exists || (ex && CompareFileTime(&ft, &s_kbm_ini_mtime) != 0)) kbm_load_cfg(0);
}

/* ---- synthetic test input (KBM_TEST) ------------------------------------- */

static struct { float t0, t1; int act; float mx, my; int wheel; int state; } s_kbm_tev[KBM_MAX_TEST];
static int s_kbm_ntev = 0;

static void kbm_parse_test(const char* e)
{
    const char* p = e;
    while (*p && s_kbm_ntev < KBM_MAX_TEST) {
        const char* q = p; while (*q && *q != ';') q++;
        char item[96]; size_t n = (size_t)(q - p); if (n >= sizeof item) n = sizeof item - 1;
        memcpy(item, p, n); item[n] = 0;
        float t0 = 0, t1 = 0; char what[64] = {0};
        if (sscanf(item, " %f - %f : %63s", &t0, &t1, what) == 3) {
            int i = s_kbm_ntev, ok = 1;
            memset(&s_kbm_tev[i], 0, sizeof s_kbm_tev[i]);
            s_kbm_tev[i].t0 = t0; s_kbm_tev[i].t1 = t1; s_kbm_tev[i].act = -1;
            char* eq = strchr(what, '=');
            if (eq) { *eq = 0;
                if (!_stricmp(what, "mouse_x"))      s_kbm_tev[i].mx = (float)atof(eq + 1);
                else if (!_stricmp(what, "mouse_y")) s_kbm_tev[i].my = (float)atof(eq + 1);
                else ok = 0;
            } else if (!_stricmp(what, "wheel_up"))   s_kbm_tev[i].wheel = 1;
            else if (!_stricmp(what, "wheel_down")) s_kbm_tev[i].wheel = -1;
            else if ((s_kbm_tev[i].act = kbm_find_action(what, strlen(what))) < 0) ok = 0;
            if (ok) s_kbm_ntev++;
            else fprintf(stderr, "[KBM] KBM_TEST: unknown item '%s'\n", what);
        } else if (n) fprintf(stderr, "[KBM] KBM_TEST: cannot parse '%s'\n", item);
        p = *q ? q + 1 : q;
    }
    fprintf(stderr, "[KBM] KBM_TEST: %d event(s); real devices %s\n", s_kbm_ntev,
            s_kbm_test_real ? "ALSO read" : "ignored, cursor never captured");
}

/* ---- window: find, subclass, capture ------------------------------------- */

static LRESULT CALLBACK kbm_game_proc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    switch (m) {
    case WM_SETCURSOR:
        if (s_kbm_captured && LOWORD(l) == HTCLIENT) { SetCursor(NULL); return TRUE; }
        break;
    case WM_KEYDOWN: case WM_KEYUP:
        /* Esc is START under KBM; Shift+Esc still reaches the backend (quit). */
        if (w == VK_ESCAPE && s_kbm_focused && s_kbm_cfg->esc_bound && !(GetKeyState(VK_SHIFT) & 0x8000))
            return 0;
        break;
    case WM_SYSCOMMAND:
        /* A bare Alt / F10 tap would enter the (invisible) system-menu loop and
         * eat the next key; Alt+Space / Alt+F4 still work (lParam != 0 / SC_CLOSE). */
        if ((w & 0xFFF0) == SC_KEYMENU && l == 0 && s_kbm_focused) return 0;
        break;
    case WM_ACTIVATE:
        if (LOWORD(w) == WA_INACTIVE && s_kbm_captured) ClipCursor(NULL);
        break;
    case WM_CLOSE: case WM_DESTROY:
        if (s_kbm_captured) ClipCursor(NULL);
        break;
    }
    return CallWindowProcA(s_kbm_prev_proc, h, m, w, l);
}

static void kbm_find_game_window(void)
{
    DWORD me = GetCurrentProcessId();
    HWND h = NULL;
    while ((h = FindWindowExA(NULL, h, "ps3recomp_d3d12", NULL)) != NULL) {
        DWORD pid = 0; GetWindowThreadProcessId(h, &pid);
        if (pid == me) break;
    }
    if (!h) return;
    s_kbm_game = h;
    if (s_kbm_sub_hwnd != h) {
        WNDPROC cur = (WNDPROC)GetWindowLongPtrA(h, GWLP_WNDPROC);
        if (cur && cur != kbm_game_proc) {
            s_kbm_prev_proc = cur;
            MemoryBarrier();
            SetWindowLongPtrA(h, GWLP_WNDPROC, (LONG_PTR)kbm_game_proc);
            s_kbm_sub_hwnd = h;
            fprintf(stderr, "[KBM] game window %p found and subclassed (cursor hide / Esc=START / Shift+Esc quits)\n", (void*)h);
            fflush(stderr);
        }
    }
}

static int kbm_client_center(POINT* c, RECT* scr)
{
    RECT r;
    if (!s_kbm_game || !GetClientRect(s_kbm_game, &r)) return 0;
    POINT tl = { r.left, r.top }, br = { r.right, r.bottom };
    ClientToScreen(s_kbm_game, &tl); ClientToScreen(s_kbm_game, &br);
    if (scr) { scr->left = tl.x; scr->top = tl.y; scr->right = br.x; scr->bottom = br.y; }
    c->x = (tl.x + br.x) / 2; c->y = (tl.y + br.y) / 2;
    return br.x > tl.x && br.y > tl.y;
}

static void kbm_clip(void)
{
    POINT c;
    if (!kbm_client_center(&c, NULL)) return;
    RECT r = { c.x, c.y, c.x + 1, c.y + 1 };
    ClipCursor(&r);
}

static int s_kbm_abs_valid = 0;
static LONG s_kbm_abs_x = 0, s_kbm_abs_y = 0;
static int s_kbm_mbtn_block = 0;

static void kbm_mlook_begin(int capture)
{
    RAWINPUTDEVICE rid; rid.usUsagePage = 1; rid.usUsage = 2; rid.dwFlags = RIDEV_INPUTSINK; rid.hwndTarget = s_kbm_msgwnd;
    if (!RegisterRawInputDevices(&rid, 1, sizeof rid))
        fprintf(stderr, "[KBM] RegisterRawInputDevices failed (err %lu): no mouse look\n", (unsigned long)GetLastError());
    InterlockedExchange(&s_kbm_mx, 0); InterlockedExchange(&s_kbm_my, 0);
    s_kbm_abs_valid = 0;
    s_kbm_mbtn_block = 1;                /* the click that focused the window is not a shot */
    s_kbm_mlook = 1;
    if (capture) {
        s_kbm_captured = 1;
        POINT c;
        if (kbm_client_center(&c, NULL)) { kbm_clip(); SetCursorPos(c.x, c.y); }
    }
    if (s_kbm_cfg->log) { fprintf(stderr, "[KBM] mouse look ON%s\n", capture ? " (cursor hidden + confined)" : " (cursor free: capture=0)"); fflush(stderr); }
}

static void kbm_mlook_end(const char* why)
{
    RAWINPUTDEVICE rid; rid.usUsagePage = 1; rid.usUsage = 2; rid.dwFlags = RIDEV_REMOVE; rid.hwndTarget = NULL;
    RegisterRawInputDevices(&rid, 1, sizeof rid);
    int was_cap = s_kbm_captured;
    s_kbm_mlook = 0; s_kbm_captured = 0;
    InterlockedExchange(&s_kbm_mx, 0); InterlockedExchange(&s_kbm_my, 0);
    if (was_cap) {
        ClipCursor(NULL);
        if (s_kbm_game) PostMessageA(s_kbm_game, WM_SETCURSOR, (WPARAM)s_kbm_game, MAKELPARAM(HTCLIENT, WM_MOUSEMOVE));
    }
    if (s_kbm_cfg->log) { fprintf(stderr, "[KBM] mouse look OFF (%s)\n", why); fflush(stderr); }
}

/* ---- raw mouse ------------------------------------------------------------ */

static int  s_kbm_wheel_acc = 0, s_kbm_wheel_n[2] = {0, 0};
static unsigned s_kbm_d_msgs = 0; static long s_kbm_d_dx = 0, s_kbm_d_dy = 0;

static void kbm_on_raw_mouse(const RAWMOUSE* m)
{
    LONG dx = 0, dy = 0;
    if (m->usFlags & MOUSE_MOVE_ABSOLUTE) {             /* RDP / VM tablets */
        int virt = (m->usFlags & MOUSE_VIRTUAL_DESKTOP) != 0;
        int w = GetSystemMetrics(virt ? SM_CXVIRTUALSCREEN : SM_CXSCREEN);
        int h = GetSystemMetrics(virt ? SM_CYVIRTUALSCREEN : SM_CYSCREEN);
        LONG ax = MulDiv(m->lLastX, w, 65535), ay = MulDiv(m->lLastY, h, 65535);
        if (s_kbm_abs_valid) { dx = ax - s_kbm_abs_x; dy = ay - s_kbm_abs_y; }
        s_kbm_abs_x = ax; s_kbm_abs_y = ay; s_kbm_abs_valid = 1;
    } else { dx = m->lLastX; dy = m->lLastY; }
    if (s_kbm_mlook) {
        if (dx || dy || m->usButtonFlags)
            InterlockedExchange64(&s_last_kbm_input_ms,
                                  (LONG64)GetTickCount64());
        if (dx) InterlockedExchangeAdd(&s_kbm_mx, dx);
        if (dy) InterlockedExchangeAdd(&s_kbm_my, dy);
        if (m->usButtonFlags & RI_MOUSE_WHEEL) {
            s_kbm_wheel_acc += (SHORT)m->usButtonData;
            while (s_kbm_wheel_acc >=  WHEEL_DELTA) { s_kbm_wheel_acc -= WHEEL_DELTA; if (s_kbm_wheel_n[0] < 3) s_kbm_wheel_n[0]++; }
            while (s_kbm_wheel_acc <= -WHEEL_DELTA) { s_kbm_wheel_acc += WHEEL_DELTA; if (s_kbm_wheel_n[1] < 3) s_kbm_wheel_n[1]++; }
        }
    }
    s_kbm_d_msgs++; s_kbm_d_dx += dx; s_kbm_d_dy += dy;
}

static LRESULT CALLBACK kbm_sink_proc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    if (m == WM_INPUT) {
        RAWINPUT ri; UINT sz = sizeof ri;
        if (GetRawInputData((HRAWINPUT)l, RID_INPUT, &ri, &sz, sizeof(RAWINPUTHEADER)) != (UINT)-1 &&
            ri.header.dwType == RIM_TYPEMOUSE)
            kbm_on_raw_mouse(&ri.data.mouse);
    }
    return DefWindowProcA(h, m, w, l);
}

/* ---- the tick: keys -> snapshot ------------------------------------------ */

static unsigned char kbm_axis(int dir, float scale)
{
    if (dir == 0) return 128;
    int v = dir > 0 ? 128 + (int)(127.0f * scale + 0.5f) : 128 - (int)(128.0f * scale + 0.5f);
    return (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
}

static int kbm_is_mouse_vk(unsigned short vk)
{
    return vk == VK_LBUTTON || vk == VK_RBUTTON || vk == VK_MBUTTON || vk == VK_XBUTTON1 || vk == VK_XBUTTON2;
}

/* snapshot layout: [0..15] pad buttons, [16..23] lx, [24..31] ly, [32..39] key-look rx,
 * [40..47] key-look ry, bit 48 left stick driven, bit 49 right stick driven by keys,
 * bit 50 KBM live (focused, or test injection running). */
static void kbm_tick(double now)
{
    static double next_cfg = 0, next_win = 0, next_clip = 0, next_log = 2.0;
    static int prev_tog = 0, prev_lb = 0, user_released = 0;
    static double wh_on_until[2] = {0, 0}, wh_next[2] = {0, 0};
    static double test_fx = 0, test_fy = 0, last_now = 0;
    static unsigned d_acts = 0; static int d_said_first = 0;
    double prev_now = last_now > 0 ? last_now : now; last_now = now;

    if (now >= next_cfg) { next_cfg = now + 1.0; kbm_cfg_check_reload(); }
    const KbmCfg* c = s_kbm_cfg;
    if ((!s_kbm_game || !IsWindow(s_kbm_game)) && now >= next_win) {
        next_win = now + 0.5; s_kbm_game = NULL; kbm_find_game_window();
    }

    int focused;
    if (s_kbm_game) focused = GetForegroundWindow() == s_kbm_game && !IsIconic(s_kbm_game);
    else            focused = pad_host_window_focused();
    if (!c->enabled || (s_kbm_test && !s_kbm_test_real)) focused = 0;
    s_kbm_focused = focused;

    /* toggle_capture edge (keyboard keys only) */
    int tog = 0;
    if (focused)
        for (int k = 0; k < KBM_MAX_KEYS; k++) {
            unsigned short vk = c->key[KA_TOGGLE][k];
            if (vk && vk < 0x100 && !kbm_is_mouse_vk(vk) && (GetAsyncKeyState(vk) & 0x8000)) tog = 1;
        }
    if (tog && !prev_tog) {
        user_released ^= 1;
        fprintf(stderr, "[KBM] %s by toggle key\n", user_released ? "mouse released" : "mouse recaptured"); fflush(stderr);
    }
    prev_tog = tog;
    /* released + left click inside the client area = recapture */
    if (focused && user_released && !s_kbm_mlook) {
        int lb = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
        if (lb && !prev_lb) {
            POINT p, cc; RECT r;
            if (GetCursorPos(&p) && kbm_client_center(&cc, &r) && PtInRect(&r, p)) user_released = 0;
        }
        prev_lb = lb;
    }
    int want = focused && !user_released;
    if (want && !s_kbm_mlook) kbm_mlook_begin(c->capture && s_kbm_game != NULL);
    else if (!want && s_kbm_mlook) kbm_mlook_end(!focused ? "focus lost" : "released");
    else if (s_kbm_mlook && (c->capture != 0) != (s_kbm_captured != 0) && s_kbm_game) {
        kbm_mlook_end("capture setting changed");      /* hot-reloaded capture= */
        kbm_mlook_begin(c->capture);
    }
    if (s_kbm_captured && now >= next_clip) { next_clip = now + 0.1; kbm_clip(); }

    unsigned act = 0;
    if (!focused && !s_kbm_test) { s_kbm_wheel_n[0] = s_kbm_wheel_n[1] = 0; }
    if (s_kbm_mbtn_block) {
        if (!(GetAsyncKeyState(VK_LBUTTON) & 0x8000) && !(GetAsyncKeyState(VK_RBUTTON) & 0x8000) &&
            !(GetAsyncKeyState(VK_MBUTTON) & 0x8000) && !(GetAsyncKeyState(VK_XBUTTON1) & 0x8000) &&
            !(GetAsyncKeyState(VK_XBUTTON2) & 0x8000)) s_kbm_mbtn_block = 0;
    }
    /* wheel notches -> pulses of wheel_ms, separated by wheel_ms/2 */
    int wh_down[2];
    for (int d = 0; d < 2; d++) {
        if (now >= wh_on_until[d] && s_kbm_wheel_n[d] > 0 && now >= wh_next[d]) {
            s_kbm_wheel_n[d]--;
            wh_on_until[d] = now + c->wheel_ms / 1000.0;
            wh_next[d] = wh_on_until[d] + c->wheel_ms / 2000.0;
        }
        wh_down[d] = now < wh_on_until[d];
    }
    if (focused) {
        int mouse_ok = s_kbm_mlook && !s_kbm_mbtn_block;
        for (int a = 0; a < KA_TOGGLE; a++) {
            for (int k = 0; k < KBM_MAX_KEYS; k++) {
                unsigned short vk = c->key[a][k];
                if (!vk) break;
                int down;
                if (vk == KVK_WHEEL_UP)        down = wh_down[0];
                else if (vk == KVK_WHEEL_DOWN) down = wh_down[1];
                else if (kbm_is_mouse_vk(vk))  down = mouse_ok && (GetAsyncKeyState(vk) & 0x8000);
                else                           down = (GetAsyncKeyState(vk) & 0x8000) != 0;
                if (down) { act |= 1u << a; break; }
            }
        }
    }
    /* KBM_TEST: synthetic actions + mouse counts, same pipeline */
    int test_live = 0;
    for (int i = 0; i < s_kbm_ntev; i++) {
        int on = now >= s_kbm_tev[i].t0 && now < s_kbm_tev[i].t1;
        if (on != s_kbm_tev[i].state) {
            s_kbm_tev[i].state = on;
            SYSTEMTIME lt; GetLocalTime(&lt);
            fprintf(stderr, "[KBM] TEST t=%.1f (%02u:%02u:%02u local) %s event %d (%s mx=%.0f my=%.0f wheel=%d)\n", now,
                    (unsigned)lt.wHour, (unsigned)lt.wMinute, (unsigned)lt.wSecond, on ? "begin" : "end", i,
                    s_kbm_tev[i].act >= 0 ? s_kbm_act_name[s_kbm_tev[i].act] : "-", s_kbm_tev[i].mx, s_kbm_tev[i].my,
                    s_kbm_tev[i].wheel);
            fflush(stderr);
            if (on && s_kbm_tev[i].wheel) s_kbm_wheel_n[s_kbm_tev[i].wheel > 0 ? 0 : 1]++;
        }
        if (!on) continue;
        test_live = 1;
        if (s_kbm_tev[i].act >= 0) act |= 1u << s_kbm_tev[i].act;
        { double lo = prev_now > s_kbm_tev[i].t0 ? prev_now : s_kbm_tev[i].t0;   /* overlap of this tick with the event */
          double span = now - lo; if (span < 0) span = 0;
          test_fx += s_kbm_tev[i].mx * span; test_fy += s_kbm_tev[i].my * span; }
    }
    if (s_kbm_ntev) {
        /* test wheel notches go through the wheel bindings even when unfocused */
        if (!focused) {
            for (int a = 0; a < KA_TOGGLE; a++)
                for (int k = 0; k < KBM_MAX_KEYS; k++) {
                    unsigned short vk = c->key[a][k];
                    if ((vk == KVK_WHEEL_UP && wh_down[0]) || (vk == KVK_WHEEL_DOWN && wh_down[1])) act |= 1u << a;
                }
        }
        LONG ix = (LONG)test_fx, iy = (LONG)test_fy;
        if (ix) { InterlockedExchangeAdd(&s_kbm_mx, ix); test_fx -= ix; }
        if (iy) { InterlockedExchangeAdd(&s_kbm_my, iy); test_fy -= iy; }
        if (wh_down[0] || wh_down[1]) test_live = 1;
    }

    unsigned btn = act & 0xFFFFu;
    if (act & ~(1u << KA_TOGGLE))
        InterlockedExchange64(&s_last_kbm_input_ms,
                              (LONG64)GetTickCount64());
    /* Keyboard movement defaults to the former Ctrl/walk deflection. Holding
     * the run binding temporarily sends full left-stick deflection. */
    float ws = (act & (1u << KA_RUN)) ? 1.0f : c->walk_scale;
    int lxd = (int)((act >> KA_LS_RIGHT) & 1) - (int)((act >> KA_LS_LEFT) & 1);
    int lyd = (int)((act >> KA_LS_DOWN) & 1)  - (int)((act >> KA_LS_UP) & 1);
    int rxd = (int)((act >> KA_RS_RIGHT) & 1) - (int)((act >> KA_RS_LEFT) & 1);
    int ryd = (int)((act >> KA_RS_DOWN) & 1)  - (int)((act >> KA_RS_UP) & 1);
    unsigned long long v = (unsigned long long)btn
        | ((unsigned long long)kbm_axis(lxd, ws) << 16) | ((unsigned long long)kbm_axis(lyd, ws) << 24)
        | ((unsigned long long)kbm_axis(rxd, c->look_keys_scale) << 32)
        | ((unsigned long long)kbm_axis(ryd, c->look_keys_scale) << 40)
        | ((unsigned long long)((lxd | lyd) != 0) << 48)
        | ((unsigned long long)((rxd | ryd) != 0) << 49)
        | ((unsigned long long)(focused || test_live) << 50);
    InterlockedExchange64(&s_kbm_pub, (LONG64)v);

    d_acts |= act & ~(1u << KA_TOGGLE);
    if (c->log && (now >= next_log || (!d_said_first && (act || s_kbm_d_msgs)))) {
        if (act || s_kbm_d_msgs) d_said_first = 1;
        if (d_acts || s_kbm_d_msgs) {
            fprintf(stderr, "[KBM] in t=%.1f: raw mouse %u msg dx=%ld dy=%ld | actions seen 0x%07X | now btn=0x%04X ls=(%u,%u) | focus=%d look=%d cap=%d%s\n",
                    now, s_kbm_d_msgs, s_kbm_d_dx, s_kbm_d_dy, d_acts, btn,
                    (unsigned)((v >> 16) & 0xFF), (unsigned)((v >> 24) & 0xFF),
                    focused, (int)s_kbm_mlook, (int)s_kbm_captured, test_live ? " TEST" : "");
            fflush(stderr);
        }
        s_kbm_d_msgs = 0; s_kbm_d_dx = s_kbm_d_dy = 0; d_acts = 0;
        next_log = now + 2.0;
    }
}

static DWORD WINAPI kbm_thread_main(LPVOID arg)
{
    (void)arg;
    WNDCLASSEXA wc; memset(&wc, 0, sizeof wc);
    wc.cbSize = sizeof wc; wc.lpfnWndProc = kbm_sink_proc;
    wc.hInstance = GetModuleHandleA(NULL); wc.lpszClassName = "ps3recomp_kbm_sink";
    RegisterClassExA(&wc);
    s_kbm_msgwnd = CreateWindowExA(0, wc.lpszClassName, "", 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, wc.hInstance, NULL);
    if (!s_kbm_msgwnd) fprintf(stderr, "[KBM] message-only window failed (err %lu): no mouse look\n", (unsigned long)GetLastError());
    while (s_kbm_run) {
        Sleep(KBM_TICK_MS);
        MSG msg;
        while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) DispatchMessageA(&msg);
        kbm_tick(kbm_now_s());
    }
    if (s_kbm_mlook) kbm_mlook_end("shutdown");
    InterlockedExchange64(&s_kbm_pub, 0);
    return 0;
}

static void kbm_atexit(void) { if (s_kbm_captured) ClipCursor(NULL); }

static void kbm_init(void)
{
    if (s_kbm_on) return;
    const char* e = getenv("KBM");
    if (e && *e == '0') { fprintf(stderr, "[KBM] disabled (KBM=0): legacy keyboard fallback\n"); return; }
    int forced = e && *e && *e != '0';
    const char* t = getenv("KBM_TEST");
    s_kbm_test = (t && *t) ? 1 : 0;
    { const char* r = getenv("KBM_TEST_REAL"); s_kbm_test_real = (r && atoi(r) > 0) ? 1 : 0; }
    if (!forced && !s_kbm_test) {
        const char* why = getenv("PAD_NO_KEYBOARD") ? "PAD_NO_KEYBOARD" :
                          (getenv("PAD_SCRIPT") && *getenv("PAD_SCRIPT")) ? "PAD_SCRIPT" :
                          (getenv("PAD_FILE") && *getenv("PAD_FILE")) ? "PAD_FILE" :
                          getenv("PAD_AUTOPRESS") ? "PAD_AUTOPRESS" : NULL;
        if (why) { fprintf(stderr, "[KBM] off: %s is set (scripted session; KBM=1 forces KBM on)\n", why); return; }
    }
    QueryPerformanceFrequency(&s_kbm_qpf); QueryPerformanceCounter(&s_kbm_t0);
    kbm_resolve_ini();
    kbm_load_cfg(1);
    if (s_kbm_test) kbm_parse_test(t);
    s_kbm_run = 1;
    s_kbm_thread = CreateThread(NULL, 0, kbm_thread_main, NULL, 0, NULL);
    if (!s_kbm_thread) { fprintf(stderr, "[KBM] thread start FAILED: legacy keyboard fallback\n"); s_kbm_run = 0; return; }
    SetThreadPriority(s_kbm_thread, THREAD_PRIORITY_ABOVE_NORMAL);
    atexit(kbm_atexit);
    s_kbm_on = 1;
    fprintf(stderr, "[KBM] keyboard+mouse active on pad port 0 (tick %d ms). F9 = release/recapture mouse, "
            "Shift+Esc = quit, KBM=0 = off\n", KBM_TICK_MS);
    fflush(stderr);
}

static void kbm_shutdown(void)
{
    if (!s_kbm_on) return;
    s_kbm_run = 0;
    WaitForSingleObject(s_kbm_thread, 200);
    CloseHandle(s_kbm_thread);
    s_kbm_thread = NULL;
    s_kbm_on = 0;
}

/* ---- pad side (pad thread / GetData): read-only on the snapshot ----------- */

/* Called from pad_poll_keyboard. merge = a real (or neutral PAD_NO_XINPUT)
 * port-0 state is already there: OR / override; else KBM owns port 0. */
static void kbm_merge(PadHostState* hs, int merge)
{
    unsigned long long v = (unsigned long long)s_kbm_pub;
    if (!merge) {
        memset(hs, 0, sizeof *hs);
        hs->connected = 1;
        hs->analog_lx = hs->analog_ly = hs->analog_rx = hs->analog_ry = 128;
    }
    if (!((v >> 50) & 1)) return;
    u16 b = (u16)(v & 0xFFFF);
    hs->buttons |= b;
    if ((v >> 48) & 1) { hs->analog_lx = (u8)(v >> 16); hs->analog_ly = (u8)(v >> 24); }
    if ((v >> 49) & 1) { hs->analog_rx = (u8)(v >> 32); hs->analog_ry = (u8)(v >> 40); }
    if (!b) return;
    if (b & CELL_PAD_CTRL_L2) hs->trigger_l2 = 255;
    if (b & CELL_PAD_CTRL_R2) hs->trigger_r2 = 255;
    if (b & CELL_PAD_CTRL_UP)       hs->press_up = 255;
    if (b & CELL_PAD_CTRL_DOWN)     hs->press_down = 255;
    if (b & CELL_PAD_CTRL_LEFT)     hs->press_left = 255;
    if (b & CELL_PAD_CTRL_RIGHT)    hs->press_right = 255;
    if (b & CELL_PAD_CTRL_TRIANGLE) hs->press_triangle = 255;
    if (b & CELL_PAD_CTRL_CIRCLE)   hs->press_circle = 255;
    if (b & CELL_PAD_CTRL_CROSS)    hs->press_cross = 255;
    if (b & CELL_PAD_CTRL_SQUARE)   hs->press_square = 255;
    if (b & CELL_PAD_CTRL_L1)       hs->press_l1 = 255;
    if (b & CELL_PAD_CTRL_R1)       hs->press_r1 = 255;
}

static unsigned char kbm_f_to_u8(float f)
{
    int v = f >= 0 ? 128 + (int)(f * 127.0f + 0.5f) : 128 - (int)(-f * 128.0f + 0.5f);
    return (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
}

/* cellPadGetData, port 0: mouse counts since the previous DELIVERED packet ->
 * right stick. fresh = this read will carry len > 0 (a new host report);
 * reads in between (the guest's drain loop) repeat the last value. */
static void kbm_mouse_to_stick(CellPadData* d, int fresh)
{
    static LARGE_INTEGER t_last;
    static float px = 0, py = 0;
    static int have = 0;
    static unsigned char rx = 128, ry = 128;
    static double d_next = 0; static unsigned d_n = 0; static long d_cx = 0, d_cy = 0; static int d_maxdev = 0;
    if (fresh) {
        const KbmCfg* c = s_kbm_cfg;
        LARGE_INTEGER now; QueryPerformanceCounter(&now);
        double dt = t_last.QuadPart ? (double)(now.QuadPart - t_last.QuadPart) / (double)s_kbm_qpf.QuadPart : 0.033;
        t_last = now;
        if (dt < 0.002) dt = 0.002;
        if (dt > 0.25)  dt = 0.25;
        LONG cx = InterlockedExchange(&s_kbm_mx, 0), cy = InterlockedExchange(&s_kbm_my, 0);
        float ox = 0, oy = 0;
        /* MOUSE (2026-10-20) mouse_carry > 0: counts are a turn DISTANCE, not just a speed.
         * The game caps the camera at full stick deflection, so a flick faster than that used to
         * be clipped (the rest of the movement was simply lost) -> the camera felt slow/short.
         * Now the part of the movement the stick could not deliver this packet stays in a bucket
         * and keeps turning on the next packets (at most mouse_carry seconds of full-speed turn),
         * so every flick turns by an amount proportional to how far the mouse moved.
         * Reversing direction drops the bucket on that axis (no overshoot). 0 = old behaviour. */
        static float pend_x = 0, pend_y = 0;
        if (c->carry > 0.0f && c->enabled) {
            if ((cx > 0 && pend_x < 0) || (cx < 0 && pend_x > 0)) pend_x = 0;
            if ((cy > 0 && pend_y < 0) || (cy < 0 && pend_y > 0)) pend_y = 0;
            pend_x += (float)cx; pend_y += (float)cy;
            const float kx = c->sens * (1.0f / 10000.0f), ky = c->sens * c->sens_y * (1.0f / 10000.0f);
            if (kx > 1e-9f && ky > 1e-9f) {   /* bucket cap: carry seconds at full deflection */
                const float capx = c->carry / kx, capy = c->carry / ky;
                if (pend_x > capx) pend_x = capx; if (pend_x < -capx) pend_x = -capx;
                if (pend_y > capy) pend_y = capy; if (pend_y < -capy) pend_y = -capy;
            }
        } else { pend_x = (float)cx; pend_y = (float)cy; }
        if ((pend_x != 0.0f || pend_y != 0.0f) && c->enabled) {
            float sx = (float)(pend_x / dt) * c->sens * (1.0f / 10000.0f);
            float sy = (float)(pend_y / dt) * c->sens * c->sens_y * (1.0f / 10000.0f);
            float m = sqrtf(sx * sx + sy * sy);
            {   /* consume what this packet delivers (linear part, capped at full deflection) */
                const float f = (m > 1.0f) ? (1.0f / m) : 1.0f;
                if (c->carry > 0.0f) {
                    pend_x -= pend_x * f; pend_y -= pend_y * f;
                    if (pend_x > -0.5f && pend_x < 0.5f) pend_x = 0;
                    if (pend_y > -0.5f && pend_y < 0.5f) pend_y = 0;
                } else { pend_x = pend_y = 0; }
            }
            if (c->invert_y) sy = -sy;
            if (m > 1e-6f) {
                float mm = m;
                if (c->accel > 0.0f) mm *= 1.0f + c->accel * mm;
                if (c->curve != 1.0f) mm = powf(mm, c->curve);
                if (mm > 1.0f) mm = 1.0f;
                float out = c->deadzone + (1.0f - c->deadzone) * mm;
                if (out > 1.0f) out = 1.0f;
                ox = sx / m * out; oy = sy / m * out;
            }
        }
        if (c->smooth > 0.0f) { ox = c->smooth * px + (1.0f - c->smooth) * ox; oy = c->smooth * py + (1.0f - c->smooth) * oy; }
        px = ox; py = oy;
        have = (ox > 0.004f || ox < -0.004f || oy > 0.004f || oy < -0.004f);
        rx = kbm_f_to_u8(ox); ry = kbm_f_to_u8(oy);
        if (have || cx || cy) {
            d_n++; d_cx += cx; d_cy += cy;
            int dev = rx > 128 ? rx - 128 : 128 - rx; int dev2 = ry > 128 ? ry - 128 : 128 - ry;
            if (dev2 > dev) dev = dev2;
            if (dev > d_maxdev) d_maxdev = dev;
        }
        double tn = kbm_now_s();
        if (c->log && d_n && tn >= d_next) {
            d_next = tn + 2.0;
            fprintf(stderr, "[KBM] pad0 t=%.1f: %u packet(s) with mouse look, counts %ld,%ld, max dev %d | last rs=(%u,%u) "
                    "(%ld,%ld counts in %.0f ms) ls=(%u,%u) btn=0x%02X%02X\n",
                    tn, d_n, d_cx, d_cy, d_maxdev, rx, ry, (long)cx, (long)cy, dt * 1000.0,
                    (unsigned)d->button[CELL_PAD_BTN_OFFSET_ANALOG_LEFT_X], (unsigned)d->button[CELL_PAD_BTN_OFFSET_ANALOG_LEFT_Y],
                    (unsigned)d->button[CELL_PAD_BTN_OFFSET_DIGITAL2], (unsigned)d->button[CELL_PAD_BTN_OFFSET_DIGITAL1]);
            fflush(stderr);
            d_n = 0; d_cx = d_cy = 0; d_maxdev = 0;
        }
    }
    if (have) {
        d->button[CELL_PAD_BTN_OFFSET_ANALOG_RIGHT_X] = rx;
        d->button[CELL_PAD_BTN_OFFSET_ANALOG_RIGHT_Y] = ry;
    }
}

/* Throttled proof that KBM buttons/left stick reach the guest packet (no mouse). */
static void kbm_note_delivered(const CellPadData* d)
{
    static double next = 0;
    unsigned long long v = (unsigned long long)s_kbm_pub;
    if (!((v >> 50) & 1) || (!(v & 0xFFFF) && !((v >> 48) & 3))) return;
    if (!s_kbm_cfg->log) return;
    double t = kbm_now_s();
    if (t < next) return;
    next = t + 2.0;
    fprintf(stderr, "[KBM] pad0 t=%.1f delivered: btn=0x%02X%02X ls=(%u,%u) rs=(%u,%u) press_cross=%u l2=%u r2=%u (kbm btn=0x%04X)\n",
            t, (unsigned)d->button[CELL_PAD_BTN_OFFSET_DIGITAL2], (unsigned)d->button[CELL_PAD_BTN_OFFSET_DIGITAL1],
            (unsigned)d->button[CELL_PAD_BTN_OFFSET_ANALOG_LEFT_X], (unsigned)d->button[CELL_PAD_BTN_OFFSET_ANALOG_LEFT_Y],
            (unsigned)d->button[CELL_PAD_BTN_OFFSET_ANALOG_RIGHT_X], (unsigned)d->button[CELL_PAD_BTN_OFFSET_ANALOG_RIGHT_Y],
            (unsigned)d->button[CELL_PAD_BTN_OFFSET_PRESS_CROSS], (unsigned)d->button[CELL_PAD_BTN_OFFSET_PRESS_L2],
            (unsigned)d->button[CELL_PAD_BTN_OFFSET_PRESS_R2], (unsigned)(v & 0xFFFF));
    fflush(stderr);
}
