# inFAMOUS 2 Reborn

A native Windows PC port of **inFamous 2** (PS3) made by static recompilation. The game's PPU
and SPU code is translated to C/C++ and compiled for x86-64, with a D3D12 renderer standing in
for the RSX.

**This repository contains no game code or game data.** The installer regenerates everything
from your own legally dumped copy of the game, then compiles it on your PC.

## Requirements

- Windows 10 or 11 (64-bit), a D3D12-capable GPU
- Your own dump of **inFamous 2, European/Australian release `BCES01143`**: a disc image
  (`.iso`, Redump images with their `.dkey` next to them) or a decrypted game folder
- About 25 GB of free space for the install; 16 GB of RAM recommended
- Python 3.10+ and Visual Studio 2022 Build Tools with the C++ and Clang tools. The setup
  program installs both through winget if they are missing.

## Install

1. Download **`inFamous 2 Setup.exe`** from the
   [Releases](https://github.com/whompay/inFAMOUS2reborn/releases) page.
2. Run it and pick your disc image or game folder.
3. Choose an install folder and let it work. The first install takes a while: the game code is
   lifted and compiled on your machine.
4. Start the game with **`inFamous 2.exe`** in the install folder. Its options window has
   V-Sync, shadows, mipmaps and button prompts.

To update, use **Update now** in the setup program or `Update inFamous 2.cmd` in your install
folder. Only changed files are downloaded and only the affected steps are redone.

## Controls

Xbox/XInput controllers work out of the box. Keyboard and mouse are supported too: bindings and
mouse sensitivity live in `infamous2-recomp\kbm.ini` (changes apply while the game runs). On-screen
prompts switch between keycaps and controller buttons depending on what you last used.

## Status

The game boots, plays its cutscenes and runs in the open world. Performance depends mostly on
CPU single-thread speed; busy scenes currently run around 20–27 FPS on the development PC.
Lighting is rendered exactly (no stale-frame halos). Expect bugs. Please report them with
`infamous2-recomp\play.err.log` attached.

## Repository layout

| Folder | What |
|---|---|
| `infamous2-recomp/` | The port: build files, entry point, input bindings, pipeline tools and the hand-written SPU registrations |
| `ps3recomp/` | The recompilation toolkit and the PS3 runtime (PPU/SPU lifters, HLE firmware libraries, D3D12 RSX backend) |
| `installer/` | The installer, setup program and launcher (Rust), the pipeline manifest `infamous2.toml`, and the patch set that turns the regenerated code into the port's maintained version |

See `installer/README.md` for how the setup is built and `infamous2-recomp/README.md` for the
port's technical notes and runtime switches.

## Credits and license

Built on the [ps3recomp](ps3recomp/README.md) toolkit (MIT; see `ps3recomp/LICENSE` and
`ps3recomp/CONTRIBUTORS.md`). Third-party components keep their own licenses under
`ps3recomp/third_party/`.

inFamous 2 is © Sony Interactive Entertainment, developed by Sucker Punch Productions. This is
an unofficial fan project, not affiliated with or endorsed by Sony or Sucker Punch. You need to
own the game to use it.
