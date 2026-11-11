/* Native registrations for inFamous 2's headerless SPURS job images.
 *
 * Each image is used in two layouts: the PPU job-chain shortcut loads it at
 * LS 0, while the WWS policy module streams it into its code buffer at LS
 * 0x4000.  Both lifts share an image id, allowing direct dispatch and overlay
 * branch dispatch to use the same content identity without address ambiguity.
 */
#include <stdint.h>
#include <stdlib.h>
#include "spu_workload.h"

extern void spu_begin_image(int image_id);
extern void spu_overlay_register_sig(const uint8_t sig[16], int image_id);

extern void job_direct_054414E2CCD626FF_spu_func_00000000(spu_context*);
extern void job_direct_054414E2CCD626FF_spu_recomp_register(void);
extern void job_overlay_054414E2CCD626FF_spu_recomp_register(void);
extern void job_direct_64D9D14299A2B7AB_spu_func_00000000(spu_context*);
extern void job_direct_64D9D14299A2B7AB_spu_recomp_register(void);
extern void job_overlay_64D9D14299A2B7AB_spu_recomp_register(void);
extern void job_direct_8927C5C0BC233071_spu_func_00000000(spu_context*);
extern void job_direct_8927C5C0BC233071_spu_recomp_register(void);
extern void job_overlay_8927C5C0BC233071_spu_recomp_register(void);
extern void job_direct_A3A153B14368B1B7_spu_func_00000000(spu_context*);
extern void job_direct_A3A153B14368B1B7_spu_recomp_register(void);
extern void job_overlay_A3A153B14368B1B7_spu_recomp_register(void);
extern void job_direct_B5EFEE742877FBAD_spu_func_00000000(spu_context*);
extern void job_direct_B5EFEE742877FBAD_spu_recomp_register(void);
extern void job_overlay_B5EFEE742877FBAD_spu_recomp_register(void);
extern void job_direct_5EC85F9D752F82D1_spu_func_00000000(spu_context*);
extern void job_direct_5EC85F9D752F82D1_spu_recomp_register(void);
extern void job_overlay_5EC85F9D752F82D1_spu_recomp_register(void);
extern void job_direct_6F43025331D3AA1A_spu_func_00000000(spu_context*);
extern void job_direct_6F43025331D3AA1A_spu_recomp_register(void);
extern void job_overlay_6F43025331D3AA1A_spu_recomp_register(void);
extern void job_direct_BCF018351EFDA059_spu_func_00000000(spu_context*);
extern void job_direct_BCF018351EFDA059_spu_recomp_register(void);
extern void job_overlay_BCF018351EFDA059_spu_recomp_register(void);
extern void job_direct_7FD662F57F44EC4C_spu_func_00000000(spu_context*);
extern void job_direct_7FD662F57F44EC4C_spu_recomp_register(void);
extern void job_overlay_7FD662F57F44EC4C_spu_recomp_register(void);
extern void job_direct_2BC127B9AAF9165A_spu_func_00000000(spu_context*);
extern void job_direct_2BC127B9AAF9165A_spu_recomp_register(void);
extern void job_overlay_2BC127B9AAF9165A_spu_recomp_register(void);

/* P172: the remaining headerless SPURS job binaries of the EBOOT job table (TOC table at
 * file 0x99C420 lists their VAs; layout 128-byte aligned between 0x889F80 and 0x8C6F80).
 * The Wwise source job at VA 0x88CB80 (6032 bytes, fp E557DC3BB8031540) was never lifted:
 * every dispatch MISSed, the job chain skipped it silently and the AK audio thread re-submitted
 * the voice forever (the title/intro "notify storm" freeze). */
extern void job_direct_315B92AC9B8B6B4C_spu_func_00000000(spu_context*);
extern void job_direct_315B92AC9B8B6B4C_spu_recomp_register(void);
extern void job_overlay_315B92AC9B8B6B4C_spu_recomp_register(void);
extern void job_direct_5025422B57A0DDDD_spu_func_00000000(spu_context*);
extern void job_direct_5025422B57A0DDDD_spu_recomp_register(void);
extern void job_overlay_5025422B57A0DDDD_spu_recomp_register(void);
extern void job_direct_34DA9231D038D0F1_spu_func_00000000(spu_context*);
extern void job_direct_34DA9231D038D0F1_spu_recomp_register(void);
extern void job_overlay_34DA9231D038D0F1_spu_recomp_register(void);
extern void job_direct_E557DC3BB8031540_spu_func_00000000(spu_context*);
extern void job_direct_E557DC3BB8031540_spu_recomp_register(void);
extern void job_overlay_E557DC3BB8031540_spu_recomp_register(void);
extern void job_direct_939A8ABE78393C5E_spu_func_00000000(spu_context*);
extern void job_direct_939A8ABE78393C5E_spu_recomp_register(void);
extern void job_overlay_939A8ABE78393C5E_spu_recomp_register(void);
extern void job_direct_4FEFED619C14A417_spu_func_00000000(spu_context*);
extern void job_direct_4FEFED619C14A417_spu_recomp_register(void);
extern void job_overlay_4FEFED619C14A417_spu_recomp_register(void);
extern void job_direct_5CDFC1EC84BDF9D0_spu_func_00000000(spu_context*);
extern void job_direct_5CDFC1EC84BDF9D0_spu_recomp_register(void);
extern void job_overlay_5CDFC1EC84BDF9D0_spu_recomp_register(void);
extern void job_direct_5EE63B107D3D252B_spu_func_00000000(spu_context*);
extern void job_direct_5EE63B107D3D252B_spu_recomp_register(void);
extern void job_overlay_5EE63B107D3D252B_spu_recomp_register(void);
extern void job_direct_D1404EE2D3742E81_spu_func_00000000(spu_context*);
extern void job_direct_D1404EE2D3742E81_spu_recomp_register(void);
extern void job_overlay_D1404EE2D3742E81_spu_recomp_register(void);
extern void job_direct_C7CA9505E88A57B6_spu_func_00000000(spu_context*);
extern void job_direct_C7CA9505E88A57B6_spu_recomp_register(void);
extern void job_overlay_C7CA9505E88A57B6_spu_recomp_register(void);
extern void job_direct_9317E059F45944BB_spu_func_00000000(spu_context*);
extern void job_direct_9317E059F45944BB_spu_recomp_register(void);
extern void job_overlay_9317E059F45944BB_spu_recomp_register(void);
extern void job_direct_10F7D3184428EC3C_spu_func_00000000(spu_context*);
extern void job_direct_10F7D3184428EC3C_spu_recomp_register(void);
extern void job_overlay_10F7D3184428EC3C_spu_recomp_register(void);
extern void job_direct_9F187A310A19C156_spu_func_00000000(spu_context*);
extern void job_direct_9F187A310A19C156_spu_recomp_register(void);
extern void job_overlay_9F187A310A19C156_spu_recomp_register(void);
extern void job_direct_E2B5E47BA5F8AFC2_spu_func_00000000(spu_context*);
extern void job_direct_E2B5E47BA5F8AFC2_spu_recomp_register(void);
extern void job_overlay_E2B5E47BA5F8AFC2_spu_recomp_register(void);
extern void job_direct_F651035BD5DA73CA_spu_func_00000000(spu_context*);
extern void job_direct_F651035BD5DA73CA_spu_recomp_register(void);
extern void job_overlay_F651035BD5DA73CA_spu_recomp_register(void);
extern void job_direct_90D94A4CAF480C0C_spu_func_00000000(spu_context*);
extern void job_direct_90D94A4CAF480C0C_spu_recomp_register(void);
extern void job_overlay_90D94A4CAF480C0C_spu_recomp_register(void);
extern void job_direct_3ED4398B76463E6D_spu_func_00000000(spu_context*);
extern void job_direct_3ED4398B76463E6D_spu_recomp_register(void);
extern void job_overlay_3ED4398B76463E6D_spu_recomp_register(void);
extern void job_direct_5238936F9B40C6A1_spu_func_00000000(spu_context*);
extern void job_direct_5238936F9B40C6A1_spu_recomp_register(void);
extern void job_overlay_5238936F9B40C6A1_spu_recomp_register(void);
extern void job_direct_1BDCAC579E2C161A_spu_func_00000000(spu_context*);
extern void job_direct_1BDCAC579E2C161A_spu_recomp_register(void);
extern void job_overlay_1BDCAC579E2C161A_spu_recomp_register(void);
extern void job_direct_3D27904E7BE0ACA5_spu_func_00000000(spu_context*);
extern void job_direct_3D27904E7BE0ACA5_spu_recomp_register(void);
extern void job_overlay_3D27904E7BE0ACA5_spu_recomp_register(void);

void infamous2_raw_jobs_register_all(void)
{
    {
        static const uint8_t sig[16] = {
            0x43,0x0C,0x88,0x02,0x42,0xB2,0xC4,0x82,
            0x43,0xA4,0x05,0x02,0x42,0x98,0xD3,0x82
        };
        spu_begin_image(300);
        job_direct_054414E2CCD626FF_spu_recomp_register();
        job_overlay_054414E2CCD626FF_spu_recomp_register();
        spu_overlay_register_sig(sig, 300);
        spu_workload_register_img(0x054414E2CCD626FFULL,
            job_direct_054414E2CCD626FF_spu_func_00000000,
            300, "infamous2_job_1040");
    }
    {
        static const uint8_t sig[16] = {
            0x43,0xDB,0xA6,0x02,0x43,0x29,0x32,0x82,
            0x42,0x33,0x53,0x02,0x42,0x3D,0x45,0x82
        };
        spu_begin_image(301);
        job_direct_64D9D14299A2B7AB_spu_recomp_register();
        job_overlay_64D9D14299A2B7AB_spu_recomp_register();
        spu_overlay_register_sig(sig, 301);
        spu_workload_register_img(0x64D9D14299A2B7ABULL,
            job_direct_64D9D14299A2B7AB_spu_func_00000000,
            301, "infamous2_job_272");
    }
    {
        static const uint8_t sig[16] = {
            0x43,0x09,0x42,0x02,0x42,0xF2,0x02,0x82,
            0x42,0x62,0x19,0x02,0x42,0x0F,0xFD,0x82
        };
        spu_begin_image(302);
        job_direct_8927C5C0BC233071_spu_recomp_register();
        job_overlay_8927C5C0BC233071_spu_recomp_register();
        spu_overlay_register_sig(sig, 302);
        spu_workload_register_img(0x8927C5C0BC233071ULL,
            job_direct_8927C5C0BC233071_spu_func_00000000,
            302, "infamous2_job_2192");
    }
    {
        static const uint8_t sig[16] = {
            0x43,0x85,0xAA,0x02,0x42,0x57,0x8E,0x82,
            0x43,0xF6,0x73,0x02,0x42,0xA5,0x2F,0x82
        };
        spu_begin_image(303);
        job_direct_A3A153B14368B1B7_spu_recomp_register();
        job_overlay_A3A153B14368B1B7_spu_recomp_register();
        spu_overlay_register_sig(sig, 303);
        spu_workload_register_img(0xA3A153B14368B1B7ULL,
            job_direct_A3A153B14368B1B7_spu_func_00000000,
            303, "infamous2_job_7696");
    }
    {
        static const uint8_t sig[16] = {
            0x43,0xE8,0x72,0x02,0x43,0x66,0x4C,0x82,
            0x42,0x15,0x07,0x02,0x42,0x1D,0xEF,0x82
        };
        spu_begin_image(304);
        job_direct_B5EFEE742877FBAD_spu_recomp_register();
        job_overlay_B5EFEE742877FBAD_spu_recomp_register();
        spu_overlay_register_sig(sig, 304);
        spu_workload_register_img(0xB5EFEE742877FBADULL,
            job_direct_B5EFEE742877FBAD_spu_func_00000000,
            304, "infamous2_job_32800");
    }
    /* Immutable descriptor/binary snapshots removed the dispatch race that
     * originally made this 18,960-byte lift appear to hang. Repeated focused
     * runs now complete full world frames, so register it unconditionally. */
    {
        static const uint8_t sig[16] = {
            0x42,0x38,0xF0,0x02,0x42,0x0D,0xD2,0x82,
            0x42,0xD2,0x25,0x02,0x42,0xF2,0x41,0x82
        };
        spu_begin_image(305);
        job_direct_5EC85F9D752F82D1_spu_recomp_register();
        job_overlay_5EC85F9D752F82D1_spu_recomp_register();
        spu_overlay_register_sig(sig, 305);
        spu_workload_register_img(0x5EC85F9D752F82D1ULL,
            job_direct_5EC85F9D752F82D1_spu_func_00000000,
            305, "infamous2_job_18960");
    }
    /* Captured after the first world transition. Its focused validation run
     * completed the first 1,584-draw world frame without a miss or deadlock. */
    {
        static const uint8_t sig[16] = {
            0x43,0xC7,0xC0,0x02,0x42,0x09,0xFC,0x82,
            0x42,0xAD,0x01,0x02,0x43,0x9F,0xFB,0x82
        };
        spu_begin_image(306);
        job_direct_6F43025331D3AA1A_spu_recomp_register();
        job_overlay_6F43025331D3AA1A_spu_recomp_register();
        spu_overlay_register_sig(sig, 306);
        spu_workload_register_img(0x6F43025331D3AA1AULL,
            job_direct_6F43025331D3AA1A_spu_func_00000000,
            306, "infamous2_job_7184");
    }
    /* Repeated world job; a focused run completed the first big frame. */
    {
        static const uint8_t sig[16] = {
            0x42,0xC2,0xD2,0x02,0x43,0x55,0xD0,0x82,
            0x43,0x6F,0xB3,0x02,0x43,0x8F,0xE9,0x82
        };
        spu_begin_image(307);
        job_direct_BCF018351EFDA059_spu_recomp_register();
        job_overlay_BCF018351EFDA059_spu_recomp_register();
        spu_overlay_register_sig(sig, 307);
        spu_workload_register_img(0xBCF018351EFDA059ULL,
            job_direct_BCF018351EFDA059_spu_func_00000000,
            307, "infamous2_job_1936");
    }
    /* This formerly shared the mutable-descriptor race above and is likewise
     * validated through the first full world frame. */
    {
        static const uint8_t sig[16] = {
            0x43,0xFB,0x86,0x02,0x42,0x31,0x0C,0x82,
            0x42,0xCC,0xB5,0x02,0x42,0x81,0x6D,0x82
        };
        spu_begin_image(308);
        job_direct_7FD662F57F44EC4C_spu_recomp_register();
        job_overlay_7FD662F57F44EC4C_spu_recomp_register();
        spu_overlay_register_sig(sig, 308);
        spu_workload_register_img(0x7FD662F57F44EC4CULL,
            job_direct_7FD662F57F44EC4C_spu_func_00000000,
            308, "infamous2_job_2192_b");
    }
    /* Revealed only after the mutable job-descriptor race was fixed; its lift
     * has now completed repeated first-world-frame validation runs. */
    {
        static const uint8_t sig[16] = {
            0x43,0xD7,0x34,0x02,0x42,0x05,0xF8,0x82,
            0x43,0xCD,0xCD,0x02,0x43,0x4C,0x03,0x82
        };
        spu_begin_image(309);
        job_direct_2BC127B9AAF9165A_spu_recomp_register();
        job_overlay_2BC127B9AAF9165A_spu_recomp_register();
        spu_overlay_register_sig(sig, 309);
        spu_workload_register_img(0x2BC127B9AAF9165AULL,
            job_direct_2BC127B9AAF9165A_spu_func_00000000,
            309, "infamous2_job_5520");
    }
    /* P172 raw jobs 310..329 (P172_RAWJOBS=0 leaves them unregistered = pre-P172 behaviour). */
    {
        const char* e = getenv("P172_RAWJOBS");
        if (e && *e == '0') { spu_begin_image(0); return; }
    }
    {   /* EBOOT VA 0x889F80 (file 0x879F80), 544 bytes extracted, trimmed 544 */
        static const uint8_t sig[16] = {
            0x42,0x03,0x3C,0x02,0x43,0x99,0x36,0x82,
            0x42,0x40,0x97,0x02,0x43,0x32,0xC3,0x82
        };
        spu_begin_image(310);
        job_direct_315B92AC9B8B6B4C_spu_recomp_register();
        job_overlay_315B92AC9B8B6B4C_spu_recomp_register();
        spu_overlay_register_sig(sig, 310);
        spu_workload_register_img(0x315B92AC9B8B6B4CULL,
            job_direct_315B92AC9B8B6B4C_spu_func_00000000,
            310, "infamous2_job_889f80");
    }
    {   /* EBOOT VA 0x88A380 (file 0x87A380), 1424 bytes extracted, trimmed 1312 */
        static const uint8_t sig[16] = {
            0x43,0x8C,0x3C,0x02,0x42,0x91,0x76,0x82,
            0x42,0x55,0xD5,0x02,0x42,0x77,0x8D,0x82
        };
        spu_begin_image(311);
        job_direct_5025422B57A0DDDD_spu_recomp_register();
        job_overlay_5025422B57A0DDDD_spu_recomp_register();
        spu_overlay_register_sig(sig, 311);
        spu_workload_register_img(0x5025422B57A0DDDDULL,
            job_direct_5025422B57A0DDDD_spu_func_00000000,
            311, "infamous2_job_88a380");
        spu_workload_register_img(0x16950FC122DF531DULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_5025422B57A0DDDD_spu_func_00000000,
            311, "infamous2_job_88a380_trim");
    }
    {   /* EBOOT VA 0x88AE00 (file 0x87AE00), 2832 bytes extracted, trimmed 2768 */
        static const uint8_t sig[16] = {
            0x42,0xA3,0xDC,0x02,0x43,0x9B,0x0A,0x82,
            0x43,0x71,0x89,0x02,0x42,0xCD,0x01,0x82
        };
        spu_begin_image(312);
        job_direct_34DA9231D038D0F1_spu_recomp_register();
        job_overlay_34DA9231D038D0F1_spu_recomp_register();
        spu_overlay_register_sig(sig, 312);
        spu_workload_register_img(0x34DA9231D038D0F1ULL,
            job_direct_34DA9231D038D0F1_spu_func_00000000,
            312, "infamous2_job_88ae00");
        spu_workload_register_img(0x6E8379B0F6B81FF1ULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_34DA9231D038D0F1_spu_func_00000000,
            312, "infamous2_job_88ae00_trim");
    }
    {   /* EBOOT VA 0x88CB80 (file 0x87CB80), 6032 bytes extracted, trimmed 5952 */
        static const uint8_t sig[16] = {
            0x43,0x53,0xFC,0x02,0x42,0x02,0xAC,0x82,
            0x43,0xE8,0xD5,0x02,0x43,0x0B,0x9B,0x82
        };
        spu_begin_image(313);
        job_direct_E557DC3BB8031540_spu_recomp_register();
        job_overlay_E557DC3BB8031540_spu_recomp_register();
        spu_overlay_register_sig(sig, 313);
        spu_workload_register_img(0xE557DC3BB8031540ULL,
            job_direct_E557DC3BB8031540_spu_func_00000000,
            313, "infamous2_job_88cb80");
        spu_workload_register_img(0x0F47E0C4B0DE8540ULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_E557DC3BB8031540_spu_func_00000000,
            313, "infamous2_job_88cb80_trim");
    }
    {   /* EBOOT VA 0x88E380 (file 0x87E380), 1936 bytes extracted, trimmed 1920 */
        static const uint8_t sig[16] = {
            0x42,0x22,0x54,0x02,0x43,0x0C,0x50,0x82,
            0x42,0x02,0xA5,0x02,0x42,0xE5,0x93,0x82
        };
        spu_begin_image(314);
        job_direct_939A8ABE78393C5E_spu_recomp_register();
        job_overlay_939A8ABE78393C5E_spu_recomp_register();
        spu_overlay_register_sig(sig, 314);
        spu_workload_register_img(0x939A8ABE78393C5EULL,
            job_direct_939A8ABE78393C5E_spu_func_00000000,
            314, "infamous2_job_88e380");
        spu_workload_register_img(0xA384BE0FDAC134DEULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_939A8ABE78393C5E_spu_func_00000000,
            314, "infamous2_job_88e380_trim");
    }
    {   /* EBOOT VA 0x890800 (file 0x880800), 5264 bytes extracted, trimmed 5200 */
        static const uint8_t sig[16] = {
            0x42,0x1A,0xF0,0x02,0x43,0x86,0x5C,0x82,
            0x43,0x05,0x21,0x02,0x43,0x24,0x6D,0x82
        };
        spu_begin_image(315);
        job_direct_4FEFED619C14A417_spu_recomp_register();
        job_overlay_4FEFED619C14A417_spu_recomp_register();
        spu_overlay_register_sig(sig, 315);
        spu_workload_register_img(0x4FEFED619C14A417ULL,
            job_direct_4FEFED619C14A417_spu_func_00000000,
            315, "infamous2_job_890800");
        spu_workload_register_img(0xF482CA32CC024D17ULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_4FEFED619C14A417_spu_func_00000000,
            315, "infamous2_job_890800_trim");
    }
    {   /* EBOOT VA 0x891D00 (file 0x881D00), 8848 bytes extracted, trimmed 8832 */
        static const uint8_t sig[16] = {
            0x42,0x26,0xF0,0x02,0x42,0xF0,0xCA,0x82,
            0x43,0xD7,0x91,0x02,0x43,0x3C,0x23,0x82
        };
        spu_begin_image(316);
        job_direct_5CDFC1EC84BDF9D0_spu_recomp_register();
        job_overlay_5CDFC1EC84BDF9D0_spu_recomp_register();
        spu_overlay_register_sig(sig, 316);
        spu_workload_register_img(0x5CDFC1EC84BDF9D0ULL,
            job_direct_5CDFC1EC84BDF9D0_spu_func_00000000,
            316, "infamous2_job_891d00");
        spu_workload_register_img(0xB3B1F55909E105D0ULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_5CDFC1EC84BDF9D0_spu_func_00000000,
            316, "infamous2_job_891d00_trim");
    }
    {   /* EBOOT VA 0x894000 (file 0x884000), 6288 bytes extracted, trimmed 6240 */
        static const uint8_t sig[16] = {
            0x43,0x20,0xDE,0x02,0x43,0x0E,0x0C,0x82,
            0x43,0xC2,0xB5,0x02,0x42,0xAE,0xE9,0x82
        };
        spu_begin_image(317);
        job_direct_5EE63B107D3D252B_spu_recomp_register();
        job_overlay_5EE63B107D3D252B_spu_recomp_register();
        spu_overlay_register_sig(sig, 317);
        spu_workload_register_img(0x5EE63B107D3D252BULL,
            job_direct_5EE63B107D3D252B_spu_func_00000000,
            317, "infamous2_job_894000");
        spu_workload_register_img(0x52BBFB3E67284CEBULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_5EE63B107D3D252B_spu_func_00000000,
            317, "infamous2_job_894000_trim");
    }
    {   /* EBOOT VA 0x896F00 (file 0x886F00), 6544 bytes extracted, trimmed 6448 */
        static const uint8_t sig[16] = {
            0x43,0x59,0xCA,0x02,0x43,0x68,0xA2,0x82,
            0x42,0xFE,0x23,0x02,0x43,0x70,0x1D,0x82
        };
        spu_begin_image(318);
        job_direct_D1404EE2D3742E81_spu_recomp_register();
        job_overlay_D1404EE2D3742E81_spu_recomp_register();
        spu_overlay_register_sig(sig, 318);
        spu_workload_register_img(0xD1404EE2D3742E81ULL,
            job_direct_D1404EE2D3742E81_spu_func_00000000,
            318, "infamous2_job_896f00");
        spu_workload_register_img(0xBD2F24EFA66D2D01ULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_D1404EE2D3742E81_spu_func_00000000,
            318, "infamous2_job_896f00_trim");
    }
    {   /* EBOOT VA 0x8A5400 (file 0x895400), 3600 bytes extracted, trimmed 3584 */
        static const uint8_t sig[16] = {
            0x43,0xF7,0xA8,0x02,0x42,0x8E,0x02,0x82,
            0x42,0xD1,0xF5,0x02,0x42,0x4A,0xCB,0x82
        };
        spu_begin_image(319);
        job_direct_C7CA9505E88A57B6_spu_recomp_register();
        job_overlay_C7CA9505E88A57B6_spu_recomp_register();
        spu_overlay_register_sig(sig, 319);
        spu_workload_register_img(0xC7CA9505E88A57B6ULL,
            job_direct_C7CA9505E88A57B6_spu_func_00000000,
            319, "infamous2_job_8a5400");
        spu_workload_register_img(0x5B4E6F453078BA36ULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_C7CA9505E88A57B6_spu_func_00000000,
            319, "infamous2_job_8a5400_trim");
    }
    {   /* EBOOT VA 0x8A6280 (file 0x896280), 3600 bytes extracted, trimmed 3568 */
        static const uint8_t sig[16] = {
            0x42,0x90,0x80,0x02,0x43,0xF9,0x08,0x82,
            0x43,0x33,0x5D,0x02,0x42,0x47,0xE9,0x82
        };
        spu_begin_image(320);
        job_direct_9317E059F45944BB_spu_recomp_register();
        job_overlay_9317E059F45944BB_spu_recomp_register();
        spu_overlay_register_sig(sig, 320);
        spu_workload_register_img(0x9317E059F45944BBULL,
            job_direct_9317E059F45944BB_spu_func_00000000,
            320, "infamous2_job_8a6280");
        spu_workload_register_img(0xD61FBB757F64E73BULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_9317E059F45944BB_spu_func_00000000,
            320, "infamous2_job_8a6280_trim");
    }
    {   /* EBOOT VA 0x8A9780 (file 0x899780), 2960 bytes extracted, trimmed 2864 */
        static const uint8_t sig[16] = {
            0x42,0xC5,0xF6,0x02,0x43,0xA7,0x26,0x82,
            0x42,0x0A,0x5B,0x02,0x42,0x0C,0x45,0x82
        };
        spu_begin_image(321);
        job_direct_10F7D3184428EC3C_spu_recomp_register();
        job_overlay_10F7D3184428EC3C_spu_recomp_register();
        spu_overlay_register_sig(sig, 321);
        spu_workload_register_img(0x10F7D3184428EC3CULL,
            job_direct_10F7D3184428EC3C_spu_func_00000000,
            321, "infamous2_job_8a9780");
        spu_workload_register_img(0xC342C6E509FB923CULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_10F7D3184428EC3C_spu_func_00000000,
            321, "infamous2_job_8a9780_trim");
    }
    {   /* EBOOT VA 0x8AA380 (file 0x89A380), 16528 bytes extracted, trimmed 16432 */
        static const uint8_t sig[16] = {
            0x42,0xF8,0x60,0x02,0x43,0x38,0xF4,0x82,
            0x43,0x86,0x4B,0x02,0x43,0x76,0xBF,0x82
        };
        spu_begin_image(322);
        job_direct_9F187A310A19C156_spu_recomp_register();
        job_overlay_9F187A310A19C156_spu_recomp_register();
        spu_overlay_register_sig(sig, 322);
        spu_workload_register_img(0x9F187A310A19C156ULL,
            job_direct_9F187A310A19C156_spu_func_00000000,
            322, "infamous2_job_8aa380");
        spu_workload_register_img(0xFCAAAEEA640C4056ULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_9F187A310A19C156_spu_func_00000000,
            322, "infamous2_job_8aa380_trim");
    }
    {   /* EBOOT VA 0x8AE480 (file 0x89E480), 19344 bytes extracted, trimmed 19216 */
        static const uint8_t sig[16] = {
            0x43,0xF6,0x8C,0x02,0x42,0x60,0x24,0x82,
            0x42,0x36,0xDB,0x02,0x43,0x4E,0xAF,0x82
        };
        spu_begin_image(323);
        job_direct_E2B5E47BA5F8AFC2_spu_recomp_register();
        job_overlay_E2B5E47BA5F8AFC2_spu_recomp_register();
        spu_overlay_register_sig(sig, 323);
        spu_workload_register_img(0xE2B5E47BA5F8AFC2ULL,
            job_direct_E2B5E47BA5F8AFC2_spu_func_00000000,
            323, "infamous2_job_8ae480");
        spu_workload_register_img(0xB5521114728C2BC2ULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_E2B5E47BA5F8AFC2_spu_func_00000000,
            323, "infamous2_job_8ae480_trim");
    }
    {   /* EBOOT VA 0x8B3080 (file 0x8A3080), 21392 bytes extracted, trimmed 21280 */
        static const uint8_t sig[16] = {
            0x42,0xA0,0xC2,0x02,0x43,0xEB,0x5C,0x82,
            0x42,0xF5,0xE5,0x02,0x43,0x0B,0x37,0x82
        };
        spu_begin_image(324);
        job_direct_F651035BD5DA73CA_spu_recomp_register();
        job_overlay_F651035BD5DA73CA_spu_recomp_register();
        spu_overlay_register_sig(sig, 324);
        spu_workload_register_img(0xF651035BD5DA73CAULL,
            job_direct_F651035BD5DA73CA_spu_func_00000000,
            324, "infamous2_job_8b3080");
        spu_workload_register_img(0x61BB9A80B936C24AULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_F651035BD5DA73CA_spu_func_00000000,
            324, "infamous2_job_8b3080_trim");
    }
    {   /* EBOOT VA 0x8B8480 (file 0x8A8480), 20624 bytes extracted, trimmed 20608 */
        static const uint8_t sig[16] = {
            0x42,0xDF,0x3A,0x02,0x43,0xD1,0x46,0x82,
            0x42,0xDF,0x6B,0x02,0x43,0x2F,0x0D,0x82
        };
        spu_begin_image(325);
        job_direct_90D94A4CAF480C0C_spu_recomp_register();
        job_overlay_90D94A4CAF480C0C_spu_recomp_register();
        spu_overlay_register_sig(sig, 325);
        spu_workload_register_img(0x90D94A4CAF480C0CULL,
            job_direct_90D94A4CAF480C0C_spu_func_00000000,
            325, "infamous2_job_8b8480");
        spu_workload_register_img(0x91724A094B48A90CULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_90D94A4CAF480C0C_spu_func_00000000,
            325, "infamous2_job_8b8480_trim");
    }
    {   /* EBOOT VA 0x8BD580 (file 0x8AD580), 23952 bytes extracted, trimmed 23904 */
        static const uint8_t sig[16] = {
            0x43,0x91,0xA6,0x02,0x43,0xD7,0x48,0x82,
            0x43,0xD3,0x0B,0x02,0x42,0xC8,0x27,0x82
        };
        spu_begin_image(326);
        job_direct_3ED4398B76463E6D_spu_recomp_register();
        job_overlay_3ED4398B76463E6D_spu_recomp_register();
        spu_overlay_register_sig(sig, 326);
        spu_workload_register_img(0x3ED4398B76463E6DULL,
            job_direct_3ED4398B76463E6D_spu_func_00000000,
            326, "infamous2_job_8bd580");
        spu_workload_register_img(0x5A7A9FC36FBEA4ADULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_3ED4398B76463E6D_spu_func_00000000,
            326, "infamous2_job_8bd580_trim");
    }
    {   /* EBOOT VA 0x8C3380 (file 0x8B3380), 7184 bytes extracted, trimmed 7088 */
        static const uint8_t sig[16] = {
            0x43,0x0A,0x84,0x02,0x42,0xDA,0x92,0x82,
            0x43,0xDB,0x03,0x02,0x42,0x30,0xF3,0x82
        };
        spu_begin_image(327);
        job_direct_5238936F9B40C6A1_spu_recomp_register();
        job_overlay_5238936F9B40C6A1_spu_recomp_register();
        spu_overlay_register_sig(sig, 327);
        spu_workload_register_img(0x5238936F9B40C6A1ULL,
            job_direct_5238936F9B40C6A1_spu_func_00000000,
            327, "infamous2_job_8c3380");
        spu_workload_register_img(0x045EF71B692D9521ULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_5238936F9B40C6A1_spu_func_00000000,
            327, "infamous2_job_8c3380_trim");
    }
    {   /* EBOOT VA 0x8C5000 (file 0x8B5000), 3600 bytes extracted, trimmed 3568 */
        static const uint8_t sig[16] = {
            0x43,0x25,0x70,0x02,0x43,0x16,0x60,0x82,
            0x42,0x48,0xB9,0x02,0x42,0x01,0x23,0x82
        };
        spu_begin_image(328);
        job_direct_1BDCAC579E2C161A_spu_recomp_register();
        job_overlay_1BDCAC579E2C161A_spu_recomp_register();
        spu_overlay_register_sig(sig, 328);
        spu_workload_register_img(0x1BDCAC579E2C161AULL,
            job_direct_1BDCAC579E2C161A_spu_func_00000000,
            328, "infamous2_job_8c5000");
        spu_workload_register_img(0x595F718DF665091AULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_1BDCAC579E2C161A_spu_func_00000000,
            328, "infamous2_job_8c5000_trim");
    }
    {   /* EBOOT VA 0x8C5E80 (file 0x8B5E80), 4240 bytes extracted, trimmed 4160 */
        static const uint8_t sig[16] = {
            0x42,0x87,0x54,0x02,0x42,0x17,0xBC,0x82,
            0x42,0x2F,0xDF,0x02,0x42,0x93,0x55,0x82
        };
        spu_begin_image(329);
        job_direct_3D27904E7BE0ACA5_spu_recomp_register();
        job_overlay_3D27904E7BE0ACA5_spu_recomp_register();
        spu_overlay_register_sig(sig, 329);
        spu_workload_register_img(0x3D27904E7BE0ACA5ULL,
            job_direct_3D27904E7BE0ACA5_spu_func_00000000,
            329, "infamous2_job_8c5e80");
        spu_workload_register_img(0x9A832799B1D13665ULL,   /* trailing-zero-trimmed content (SPU_JOB_TRIM_MATCH) */
            job_direct_3D27904E7BE0ACA5_spu_func_00000000,
            329, "infamous2_job_8c5e80_trim");
    }
    spu_begin_image(0);
}

__attribute__((constructor)) static void infamous2_raw_jobs_register_ctor(void)
{
    infamous2_raw_jobs_register_all();
}
