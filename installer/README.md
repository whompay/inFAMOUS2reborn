# infamous2-installer

Builds the inFamous 2 (BCES01143) PC port from the player's **own** copy of the game. The port's
published tree holds no game code and no game data. The installer regenerates all of it from the
player's dump, applies the port's patches, compiles it and writes a launcher.

This is the inFamous installer (`..\..\inFamous\installer`) set up for inFamous 2: same program,
same setup window and same update mechanism, with its own manifest (`infamous2.toml`), binaries
(`infamous2-installer`, `infamous2-launch`, `infamous2-setup`) and launcher options.

## Making the setup (maintainer)

Double-click **`make-setup.cmd`** in the `inFamous 2` folder. It:

1. builds the installer (`cargo build --release` in `installer\`),
2. runs `infamous2-installer doctor` (Python 3.10+, VS 2022 Build Tools with clang-cl),
3. runs `make-patches` against your dump in `Infamous 2 (Europe, Australia) (...)`,
4. runs `package` → **`dist\inFamous 2 Setup.exe`**.

`make-patches` exports a clean tree (exactly what a player gets) to `.installer\fresh`, runs the
whole pipeline there, compares the output with your `infamous2-recomp` and writes
`installer\generated.lock` and `installer\patches\`. It doesn't change any of your files. The PPU
lift is the long part. Extra arguments to `make-setup.cmd` go to `make-patches`: `--keep-fresh`
reuses the exported tree, `--regen` re-runs every step.

Run it again whenever the lifted files in `infamous2-recomp` change (a new lift, hand edits to
`recompiled/` or `spu/`), so the patch set matches your tree.

### When make-patches stops

It refuses to finish if a patch would amount to publishing generated content:

* **A binary file differs from the pipeline's output**, or the pipeline doesn't produce a file
  you have. A step is missing or uses different arguments than you used. Fix the step in
  `infamous2.toml` and run it again.
* **A patch carries more than half of a file.** That means the file is mostly not what the
  pipeline makes, so the patch would ship lifted code. Your `ppu_lifter.py` hasn't changed since
  `recompiled/` was lifted (both 2026-10-11), so the PPU units should come out close to your files.
  If the toolkit changes later, re-lift your tree (`lift.cmd`), rebuild, test, then run
  `make-setup.cmd` again. `--allow-leaks` accepts the findings knowingly.

## The pipeline (what runs on a player's PC)

`infamous2.toml` is `lift.cmd`, step by step:

| Step | What |
|---|---|
| ppu-loader, ppu-lift | functions/imports of EBOOT.elf, then the 8 lifted PPU units |
| spu-images, spu-lift | the 2 embedded SPU ELFs and `spu_workloads.c` |
| edgezlib | `tools/patch_edgezlib_worker.py` on the EDGE-zlib worker lift |
| spu-regcache | register-cached twin of `spu_0001_at_0085E400` |
| carve | `tools/carve_blobs.py`: policy module, 30 raw SPURS jobs, 27 WWS job modules, 5 plugins |
| pm-functions, pm-lift | the WWS policy module (`spu/pm_wwsjob`) |
| raw-jobs-lift | `tools/lift_raw_jobs.py`: each raw job at LS 0 and LS 0x4000 |
| jobmods-lift | `lift_jobmods.py` with `spu/jobmods_extra_funcs.json` |
| wwsplug | `plug_lift.py` with `spu/fixed_regions.json` |

Two new tools in `infamous2-recomp/tools/` make this reproducible from a disc alone:

* **`carve_blobs.py`** cuts every SPU blob out of EBOOT.elf by offset and checks each against the
  SHA-1 of your file. All 27 modules in `spu/jobmods` turned out to be byte ranges of the EBOOT,
  including the ones first captured at run time. One, `jobmod_49d0bbde9a83_e2060`, carries
  12 bytes the job had written into its own data words before the capture. The script re-applies
  them, so the SHA-1 and content signature match your corpus. It also seeds
  `spu/jobmods_manifest.json` with your module order, because `lift_jobmods.py` numbers images
  200+ in manifest order.
* **`lift_raw_jobs.py`** is the `for %%J` loop of `lift.cmd`.

`spu/fixed_regions.json`, `spu/jobmods_extra_funcs.json`, `spu/pm_wwsjob_register.c` and
`spu/raw_jobs_register.c` are hand-written. They're excluded from `[generated]` and published as
they are.

## Build and launch

The build uses `-DPORT_PPU_REGCACHE_ONLY=ppu_recomp_001`, the SPU register-cache twin of the
lighting job module (`-DPORT_SPU_RC_GEN=ON -DPORT_SPU_RC_ONLY=jobmod_infamous2_batch_e5400
-DPORT_SPU_RC_XFLOAT_FIXED=ON`) and PGO off (PGO profiles are tied to one LLVM version) in its own `infamous2-recomp\build-install`.
Your `build-fast` / `build-pgo-*` trees are never touched. Players get the non-PGO build, so
expect a few FPS less than your `build-pgo-use23` exe.

`inFamous 2.exe` (the launcher) starts the game with the environment of `run.cmd` (profilers,
captures and bring-up logging off, `BLIT_SYNC=1` + `BLIT_DEFER=1`, `ZCONV_STENCIL=1`,
`SPURS_PCORE_WIDS=2` and the performance switches listed in `[launch]`).
Its options window has only what the inFamous 2 runtime reads: V-Sync (`PRESENT_VSYNC`),
Shadows (`SHADOWS`), Mipmaps (`TEX_MIPS`) and Button Prompts (`KBM_GLYPHS`). The port has no
in-game Settings menu, so the launcher saves these in `infamous2-recomp\settings.ini` and passes
them to the game as environment variables. Key bindings and mouse sensitivity stay in
`infamous2-recomp\kbm.ini`.

## What gets published

`export` (and the setup) contain `ps3recomp/`, `installer/` and, from `infamous2-recomp/`, only
the allow-list in `[export] keep`: CMakeLists, `main.cpp`, `stubs.cpp`, `spu_plug_stub.c`,
`config.toml`, `kbm.ini`, `README.md`, `build.cmd`, the pipeline tools and the four hand-written
SPU files. Never published (`forbid` makes `export`/`package` stop if they would be):

* captures and memory dumps (`*.bin` in the port folder, `rsxcap*`, `capture*`, `spu-miss*`,
  `variant-capture`),
* save data (`gamedata`), `test-runs`, `shadercache`, `pgo-data`, `_jobs`, `_xfer`, maintainer
  notes, the playbooks, scratch output folders,
* `ps3recomp/fw_spu/`, which holds Sony firmware modules (`libsre.prx`, `libatxdec.prx`, ...) and
  code lifted from them. The inFamous 2 build doesn't use them.

`infamous2-installer export --out <empty dir>` shows the exact tree. Review it before pushing.

## Updates

`[update] repo = "whompay/inFAMOUS2reborn"`. Push the `export` tree there (generated.lock and
patches included). Players then use **Update now** in the setup program, or `Update inFamous 2.cmd` in
their install folder, to get only the changed files. If the repository is private, the updater needs a `GITHUB_TOKEN`, so
send players a new `inFamous 2 Setup.exe` instead. Running it over an existing install updates it
and only redoes what changed.

## For players

Run `inFamous 2 Setup.exe`. Then:

1. Pick the disc image (.iso) or a game folder (BCES01143). Redump images work with their
   `.dkey` next to them.
2. Let it install Python and the Visual Studio C++ build tools if they're missing.
3. Choose an install folder. Allow about 25 GB plus the port, and 16 GB of RAM helps.

The first install takes a while: the game code is lifted and compiled on your PC.
