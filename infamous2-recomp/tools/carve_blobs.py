#!/usr/bin/env python3
"""carve_blobs.py -- cut the SPU code the port lifts out of the player's own EBOOT.elf.

Used by the installer (installer/infamous2.toml, step "carve") and equivalent to the
extract_blob.py calls in lift.cmd, plus the modules lift.cmd did not carve because they were
first captured at run time (all of them turned out to be byte ranges of the EBOOT).

    python carve_blobs.py <EBOOT.elf> <infamous2-recomp dir>

Every output is checked against the SHA-1 of the file the port was built from. The table holds
offsets, lengths and hashes only -- no game bytes.

One module, jobmod_49d0bbde9a83_e2060, was captured from local store after the job had run: its
last 0x20 bytes carry twelve bytes the job writes into its own data area. They are re-applied
below (RUNTIME_WORDS) so the file -- and the content signature the runtime matches it by -- is
identical to the captured corpus.

It also seeds spu/jobmods_manifest.json with the module order of the maintained tree.
lift_jobmods.py numbers the job-module images in the order of an existing manifest, and the port's
hand-written registrations depend on those image ids (200 + index).
"""
import hashlib
import json
import os
import sys

# (output relative to infamous2-recomp, EBOOT.elf file offset, length, sha1 of the result)
BLOBS = [
    # WWS job manager policy module (guest VA 0x887100; first PT_LOAD at VA 0x10000 / file 0)
    ("spu-images/pm_wwsjob.raw", 0x877100, 0x2E80, "50dc1c2cb3f84e03ee2f62f56c203f34b5bbdad4"),
    # headerless SPURS job binaries of the EBOOT job table
    ("spu-images/raw-jobs/job_054414E2CCD626FF.bin", 0x87A980, 0x410, "9d2ce29bc618f7f4c4e1431a7024a3e13760ddb2"),
    ("spu-images/raw-jobs/job_10F7D3184428EC3C.bin", 0x899780, 0xB90, "435806eceb809796a9c382cfd17d585b18d0b9dc"),
    ("spu-images/raw-jobs/job_1BDCAC579E2C161A.bin", 0x8B5000, 0xE10, "0900a3bd96f9bea4a05fa19fbee80d071f5d5833"),
    ("spu-images/raw-jobs/job_2BC127B9AAF9165A.bin", 0x885900, 0x1590, "6119af7a44820ea9e6e2a6f31414b9cccd425000"),
    ("spu-images/raw-jobs/job_315B92AC9B8B6B4C.bin", 0x879F80, 0x220, "925fe7180731f23b79f4fff11d0cc38595febb41"),
    ("spu-images/raw-jobs/job_34DA9231D038D0F1.bin", 0x87AE00, 0xB10, "9f7276f8ef09797142c407faaa7ccdb1f55cf526"),
    ("spu-images/raw-jobs/job_3D27904E7BE0ACA5.bin", 0x8B5E80, 0x1090, "0b316e672681c2e8090ba667e923e158b8dd2e71"),
    ("spu-images/raw-jobs/job_3ED4398B76463E6D.bin", 0x8AD580, 0x5D90, "526aafcf14d792c0c08e3b86369b67ee5e3a78e8"),
    ("spu-images/raw-jobs/job_4FEFED619C14A417.bin", 0x880800, 0x1490, "281a845fedc3a1965bf6e1994da2055f944f970f"),
    ("spu-images/raw-jobs/job_5025422B57A0DDDD.bin", 0x87A380, 0x590, "bd76df821d7bc6746e605442920cbea4f5ceb760"),
    ("spu-images/raw-jobs/job_5238936F9B40C6A1.bin", 0x8B3380, 0x1C10, "4cf3b6a70337e08a433342c60ff73cbe86df4102"),
    ("spu-images/raw-jobs/job_5CDFC1EC84BDF9D0.bin", 0x881D00, 0x2290, "149e834eb38fd13df6254693c215ce4072b00614"),
    ("spu-images/raw-jobs/job_5EC85F9D752F82D1.bin", 0x890980, 0x4A10, "936df9a01faa88b5ec0108749611b754a1062a72"),
    ("spu-images/raw-jobs/job_5EE63B107D3D252B.bin", 0x884000, 0x1890, "1079f910d7c8377e09986d3ec1bcb89008536382"),
    ("spu-images/raw-jobs/job_64D9D14299A2B7AB.bin", 0x87A200, 0x110, "46568a8f2afa3a6bef417e769c545ad2f3d2eedb"),
    ("spu-images/raw-jobs/job_6F43025331D3AA1A.bin", 0x87EB80, 0x1C10, "37c2a63a54c6e012e20492ff1539a99989063b31"),
    ("spu-images/raw-jobs/job_7FD662F57F44EC4C.bin", 0x87B980, 0x890, "3bd5d6041fb240cc4da8bb876e333844f81ad782"),
    ("spu-images/raw-jobs/job_8927C5C0BC233071.bin", 0x87C280, 0x890, "c37a7ea52794e10a321135b7552209d3733e7833"),
    ("spu-images/raw-jobs/job_90D94A4CAF480C0C.bin", 0x8A8480, 0x5090, "0ed1009baffecb1f7c08fd2ae6003f1723f15d3b"),
    ("spu-images/raw-jobs/job_9317E059F45944BB.bin", 0x896280, 0xE10, "3bab2a92973c0e9dc8116484c6fc029b28e16a2c"),
    ("spu-images/raw-jobs/job_939A8ABE78393C5E.bin", 0x87E380, 0x790, "657f399512005c46746b4a494cbe917fac31dc6b"),
    ("spu-images/raw-jobs/job_9F187A310A19C156.bin", 0x89A380, 0x4090, "72fc4e099c2f663faf98741702016ce4a956bbc7"),
    ("spu-images/raw-jobs/job_A3A153B14368B1B7.bin", 0x897100, 0x1E10, "a410c85ff398f8bad073a0e5faeeb2302882ec1e"),
    ("spu-images/raw-jobs/job_B5EFEE742877FBAD.bin", 0x888900, 0x8020, "3faf5cd7fb07be029f3370a651d702067825144e"),
    ("spu-images/raw-jobs/job_BCF018351EFDA059.bin", 0x898F80, 0x790, "c3bfbeed9badb0e1509fc4e49cc828cc955e203f"),
    ("spu-images/raw-jobs/job_C7CA9505E88A57B6.bin", 0x895400, 0xE10, "4ab34c3bc10d8555d8ad4c935836a9b0a4feef1c"),
    ("spu-images/raw-jobs/job_D1404EE2D3742E81.bin", 0x886F00, 0x1990, "818cdea8bf557bf29e70a0170cfe8ed6dc90fd87"),
    ("spu-images/raw-jobs/job_E2B5E47BA5F8AFC2.bin", 0x89E480, 0x4B90, "83db2fc2db827434bd3998b09edc750bccb7fbb9"),
    ("spu-images/raw-jobs/job_E557DC3BB8031540.bin", 0x87CB80, 0x1790, "882b554d853a119d5e2902d832c633937ab32d22"),
    ("spu-images/raw-jobs/job_F651035BD5DA73CA.bin", 0x8A3080, 0x5390, "264880b2f7643848054ae3f04545d2713fce86f6"),
    # headered WWS job modules (streamed into the job code buffer at LS 0x4000)
    ("spu/jobmods/jobmod_0b71755d6f4a_e5A00.bin", 0x848C80, 0x61C0, "4b472e14a4d8ae30f5e7ab03273403ed5237feb3"),
    ("spu/jobmods/jobmod_1d732ac17279_e61A0.bin", 0x851A00, 0x6630, "1d732ac172795e47f6881166bbdf43b686ee916c"),
    ("spu/jobmods/jobmod_43398b961dfa_e1390.bin", 0x84EE80, 0x17C0, "43398b961dfa20b8339c3512d2748c45e67e5e06"),
    ("spu/jobmods/jobmod_46e77f5a9167_e3E00.bin", 0x7C3800, 0x4370, "46e77f5a91676dff06b84d9d2d03ed371c3825be"),
    ("spu/jobmods/jobmod_49d0bbde9a83_e2060.bin", 0x80C980, 0x2330, "49d0bbde9a8343d1432b4f6b1e5ce68db206671b"),
    ("spu/jobmods/jobmod_4ccdc05864ab_e9808.bin", 0x7E7900, 0x9DF0, "4ccdc05864ab76f895e086f34983c08b4208ad50"),
    ("spu/jobmods/jobmod_512689a33f01_eC7C0.bin", 0x7F2100, 0xEB00, "512689a33f0126be3bd5c4283151d338672557ad"),
    ("spu/jobmods/jobmod_638a73a6476e_e60C0.bin", 0x7BA480, 0x65F0, "638a73a6476e2bd96f53403e91a6baa5bc027ed9"),
    ("spu/jobmods/jobmod_6763851e4f29_e1980.bin", 0x7E3500, 0x1E50, "6763851e4f2953b3994904bc5f2336c6ddfc114c"),
    ("spu/jobmods/jobmod_6c2ac7b491f5_eFE70.bin", 0x81E800, 0x1A9A0, "2025343b09d0391a442af43eac5ac5578ab3eff5"),
    ("spu/jobmods/jobmod_74aa05413402_e2F30.bin", 0x7B5480, 0x4FD0, "74aa05413402c80c936047a82d43949d0b2dd39a"),
    ("spu/jobmods/jobmod_74aad00cc5f6_e2118.bin", 0x7E5380, 0x2560, "74aad00cc5f68e5b02665e1d5e463e1d7dc9d878"),
    ("spu/jobmods/jobmod_867e5d2348b3_e2898.bin", 0x7C0A80, 0x2D60, "867e5d2348b37c5d51c3ad041411b0d7fc9f4996"),
    ("spu/jobmods/jobmod_87d401776ff2_e6CE0.bin", 0x7A4E80, 0x79A0, "87d401776ff2aebc62eda72f8cd34f02b2c5b767"),
    ("spu/jobmods/jobmod_956917938376_e2E98.bin", 0x85B200, 0x31D0, "956917938376e43809c44b8dab411bc007948ea8"),
    ("spu/jobmods/jobmod_979cbcdc6dd8_e58A0.bin", 0x7DD680, 0x5EA0, "979cbcdc6dd82cc44b34fb61b303651def08e7f1"),
    ("spu/jobmods/jobmod_986b36f10d14_e4640.bin", 0x800C00, 0x4940, "e9809289d7f388fc33b58677d5589b65d05fb094"),
    ("spu/jobmods/jobmod_9c3ccc44b507_e10E8.bin", 0x850680, 0x1340, "9c3ccc44b5076d0769a63afecff4185e2ca03105"),
    ("spu/jobmods/jobmod_9c911ed91cc0_e1200.bin", 0x805580, 0x1740, "9c911ed91cc05ad563b4f28c04627f279b6af3ec"),
    ("spu/jobmods/jobmod_b7507967da11_e0760.bin", 0x7F1700, 0x9C0, "b7507967da11df49343e629f3730c49995cdb393"),
    ("spu/jobmods/jobmod_cd7019d4687a_e0808.bin", 0x81DD00, 0xAC0, "cd7019d4687a6264e5de37d225ae6b79ae726cbe"),
    ("spu/jobmods/jobmod_d0d15ec83992_eA2A0.bin", 0x812600, 0xB6D0, "d0d15ec83992cbafbe9f05714cb4266edff17d5e"),
    ("spu/jobmods/jobmod_e094d9aca2b5_eF1F0.bin", 0x839200, 0xFA60, "e094d9aca2b5b249b9329f3c8afb4851bd94330f"),
    ("spu/jobmods/jobmod_e13f67defd8a_e2BC0.bin", 0x858080, 0x3190, "e13f67defd8a4b08c984346548d7400e8ff2b4ae"),
    ("spu/jobmods/jobmod_f45bee96696d_e83E0.bin", 0x7AC800, 0x8C30, "f45bee96696dbaea1aca79427069eeeacf64cd79"),
    ("spu/jobmods/jobmod_infamous2_batch_e5400.bin", 0x806D00, 0x5C60, "9a537c60d338a6bb21159f47c146c3d1192c53f2"),
    ("spu/jobmods/jobmod_infamous2_e14f90.bin", 0x7C7B80, 0x15B10, "65311004ca94a542773ee18c3dd4d90f9dc25145"),
    # streamed plugins observed at LS 0x6C00 (embedded ELF bytes)
    ("spu-images/plugin_6c00_0790.bin", 0x80FC80, 0x790, "fd0bbb84bb4e1121978328dbbf8eacf8f3066e04"),
    ("spu-images/plugin_6c00_0b90.bin", 0x80F080, 0xB90, "1d41fa47e63448e461bcb513b96b5669f14b1c71"),
    ("spu-images/plugin_6c00_0b90_alt.bin", 0x810480, 0xB90, "49187bc8d92832752fd19704fc3d0b02d6b9ad6b"),
    ("spu-images/plugin_6c00_1510.bin", 0x811080, 0x1510, "8e4b93671cb8a3ead39e1f5eef4fb5cfe0234728"),
    ("spu-images/plugin_6c00_6c30.bin", 0x80ED00, 0x310, "48107d6aa139f090966876c1e4c43faa0fcb7d54"),
]

# Bytes the job writes into its own data words before it was captured (offset -> value).
RUNTIME_WORDS = {
    "spu/jobmods/jobmod_49d0bbde9a83_e2060.bin": {
        0x2312: 0x13, 0x2313: 0x30, 0x2316: 0x13, 0x2317: 0x30,
        0x231A: 0x13, 0x231B: 0x30, 0x231E: 0x13, 0x231F: 0x30,
        0x2323: 0x01, 0x2327: 0x01, 0x232B: 0x01, 0x232F: 0x01,
    },
}

# Image order of the maintained tree (lift_jobmods.py: image id = 200 + index).
JOBMOD_ORDER = [
    "jobmod_0b71755d6f4a_e5A00", "jobmod_6c2ac7b491f5_eFE70", "jobmod_986b36f10d14_e4640",
    "jobmod_infamous2_batch_e5400", "jobmod_49d0bbde9a83_e2060", "jobmod_infamous2_e14f90",
    "jobmod_1d732ac17279_e61A0", "jobmod_43398b961dfa_e1390", "jobmod_46e77f5a9167_e3E00",
    "jobmod_4ccdc05864ab_e9808", "jobmod_512689a33f01_eC7C0", "jobmod_638a73a6476e_e60C0",
    "jobmod_6763851e4f29_e1980", "jobmod_74aa05413402_e2F30", "jobmod_74aad00cc5f6_e2118",
    "jobmod_867e5d2348b3_e2898", "jobmod_87d401776ff2_e6CE0", "jobmod_956917938376_e2E98",
    "jobmod_979cbcdc6dd8_e58A0", "jobmod_9c3ccc44b507_e10E8", "jobmod_9c911ed91cc0_e1200",
    "jobmod_b7507967da11_e0760", "jobmod_cd7019d4687a_e0808", "jobmod_d0d15ec83992_eA2A0",
    "jobmod_e094d9aca2b5_eF1F0", "jobmod_e13f67defd8a_e2BC0", "jobmod_f45bee96696d_e83E0",
]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    elf_path, port = sys.argv[1], sys.argv[2]
    elf = open(elf_path, "rb").read()
    bad = []
    for rel, off, length, sha in BLOBS:
        if off + length > len(elf):
            bad.append(f"{rel}: 0x{off:X}+0x{length:X} is past the end of {elf_path}")
            continue
        data = bytearray(elf[off:off + length])
        for at, val in RUNTIME_WORDS.get(rel, {}).items():
            data[at] = val
        got = hashlib.sha1(data).hexdigest()
        if got != sha:
            bad.append(f"{rel}: sha1 {got}, expected {sha}")
            continue
        dst = os.path.join(port, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # Leave an identical file alone so its timestamp (and the build) stays as it was.
        try:
            if open(dst, "rb").read() == data:
                continue
        except OSError:
            pass
        with open(dst, "wb") as f:
            f.write(data)
    if bad:
        print("This EBOOT.elf does not contain the SPU code the port was made from:", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        return 1
    seed = os.path.join(port, "spu", "jobmods_manifest.json")
    if not os.path.exists(seed):
        with open(seed, "w") as f:
            json.dump([{"name": n} for n in JOBMOD_ORDER], f, indent=2)
    print(f"carved {len(BLOBS)} SPU blobs from {elf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
