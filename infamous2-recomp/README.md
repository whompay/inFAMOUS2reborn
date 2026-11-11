# inFamous 2 — native PC port (PS3 static recompilation)

Native recompilation of the European/Australian disc release `BCES01143`, built on the
`ps3recomp` toolkit beside this directory. No game code or data is stored here: the lifted
PPU/SPU code is regenerated from your own disc dump by the installer.

## Status

- The game boots, plays cutscenes and runs in the open world on Windows 10/11 through a
  D3D12 renderer, with save data, audio and Bink movies.
- PPU: 24,899 recovered functions (30,918 entries including mid-function targets) lifted
  into eight translation units, 257 firmware imports across 19 libraries.
- SPU: both embedded SPU images, the WWS policy module, all 30 raw SPURS jobs, 27 WWS job
  modules and 5 streamed plugins are lifted and registered by content identity. The SPU
  interpreter is essentially unused during play.
- Deferred lighting is exact: the SPU lighting job reads same-frame depth/normal copies, and
  the GPU readback overlaps other work so the wait is paid only when the job consumes it.
- Keyboard and mouse are supported alongside XInput controllers (see below).

Performance depends heavily on the CPU (the PPU main thread is the critical path). On the
development PC busy open-world scenes run at roughly 20–27 FPS, lighter areas higher.

## Build and run (from source)

Most players should use the installer (see the repository README): it regenerates the lifted
code from your dump, builds it and writes the `inFamous 2.exe` launcher.

To build by hand after the installer has generated `recompiled/` and `spu/`: from a Visual Studio
2022 developer command prompt with clang-cl and Python 3.10+ on `PATH` (or `PYTHON` set to a
python.exe), run `build.cmd`. Start the game with
`inFamous2Recomp.exe <path to PS3_GAME\USRDIR\EBOOT.elf> <path to the disc dump>`.

### Runtime switches

The launcher sets the release defaults (`[launch]` in `installer/infamous2.toml`). Useful overrides:

| Variable | Default | Meaning |
|---|---|---|
| `BLIT_SYNC` / `BLIT_DEFER` | `1` / `1` | Exact same-frame lighting with an overlapped readback. `BLIT_SYNC=0` is faster but can show dark halos during fast motion. |
| `RSX_RENDER_PCORE` | `1` | Prefer performance cores for the D3D12 replay worker on hybrid CPUs. |
| `SPURS_PCORE_WIDS` | `2` | Pin the frame-critical SPU workload to performance cores (`-` disables). |
| `FIFO_KICK_LOOP` | `1` | Event-driven RSX FIFO drain instead of a fixed sleep. |
| `SPU_DMA_OPT` | `F` | Cheaper SPU DMA paths (bit mask). |
| `SPURS_SPIN_MODE` | `pause` | SPURS kernel idle spin uses `pause`. |
| `PRESENT_VSYNC`, `SHADOWS`, `TEX_MIPS`, `KBM_GLYPHS` | | The launcher's options window sets these. |

## Input

Keyboard and mouse map onto the DualShock 3 on pad port 0. Bindings and mouse sensitivity are
in `kbm.ini` (hot-reloaded). Keyboard movement walks by default; hold Left Shift to run
(`walk_scale` in `kbm.ini`).

The game's contextual prompts show keycaps during keyboard/mouse play: the renderer substitutes
the controller-glyph atlas cells in place, following the live bindings. Controller input switches
back to the original glyphs immediately; stick drift inside the deadzone does not. Set
`KBM_GLYPHS=0` to disable substitution.

## Regenerate the lift

The pipeline is the `[[step]]` list in `installer/infamous2.toml`: load the decrypted ELF, lift
the PPU code, extract and lift the embedded SPU images, carve and lift the policy module, SPURS
jobs, job modules and plugins, then regenerate the workload registries. `infamous2-installer install --regen`
re-runs every step.
