//! The user's game files: locating the dump, reading PARAM.SFO, getting a decrypted EBOOT.elf,
//! and the byte-level extractions the pipeline needs.

use crate::bail;
use crate::hash;
use crate::util::{self, Ctx, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct Game {
    /// Folder that holds PS3_GAME (and usually PS3_DISC.SFB).
    pub root: PathBuf,
    pub usrdir: PathBuf,
    pub sfo: BTreeMap<String, String>,
}

impl Game {
    pub fn title_id(&self) -> &str {
        self.sfo.get("TITLE_ID").map(|s| s.as_str()).unwrap_or("?")
    }
    pub fn title(&self) -> &str {
        self.sfo.get("TITLE").map(|s| s.as_str()).unwrap_or("?")
    }
    pub fn eboot_bin(&self) -> PathBuf {
        self.usrdir.join("EBOOT.BIN")
    }
    pub fn eboot_elf(&self) -> PathBuf {
        self.usrdir.join("EBOOT.elf")
    }
}

/// Accepts the disc folder, its PS3_GAME or USRDIR folder, or the EBOOT file itself.
pub fn locate(input: &Path) -> Result<Game> {
    let input = util::absolute(input);
    let mut cands: Vec<PathBuf> = Vec::new();
    let mut p: Option<&Path> = Some(&input);
    for _ in 0..4 {
        let Some(d) = p else { break };
        cands.push(d.to_path_buf());
        p = d.parent();
    }
    for c in &cands {
        let sfo = c.join("PS3_GAME").join("PARAM.SFO");
        if sfo.is_file() {
            let usrdir = c.join("PS3_GAME").join("USRDIR");
            if !usrdir.is_dir() {
                bail!("{} has no PS3_GAME\\USRDIR folder", c.display());
            }
            let sfo = read_sfo(&sfo)?;
            return Ok(Game {
                root: c.clone(),
                usrdir,
                sfo,
            });
        }
    }
    bail!(
        "could not find PS3_GAME\\PARAM.SFO in or above {}.\n    Point the installer at the folder of your game dump \
         (the one that contains PS3_GAME).",
        input.display()
    )
}

pub fn read_sfo(path: &Path) -> Result<BTreeMap<String, String>> {
    let d = fs::read(path).ctx(|| format!("reading {}", path.display()))?;
    parse_sfo(&d).ctx(|| format!("{} is not a valid PARAM.SFO", path.display()))
}

fn u16le(d: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(d.get(o..o + 2)?.try_into().ok()?))
}
fn u32le(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?))
}
fn u32be(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_be_bytes(d.get(o..o + 4)?.try_into().ok()?))
}
fn u16be(d: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_be_bytes(d.get(o..o + 2)?.try_into().ok()?))
}

pub fn parse_sfo(d: &[u8]) -> Option<BTreeMap<String, String>> {
    if d.get(0..4)? != b"\0PSF" {
        return None;
    }
    let key_tab = u32le(d, 8)? as usize;
    let data_tab = u32le(d, 12)? as usize;
    let n = u32le(d, 16)? as usize;
    let mut out = BTreeMap::new();
    for i in 0..n.min(1024) {
        let e = 20 + i * 16;
        let key_off = u16le(d, e)? as usize;
        let fmt = u16le(d, e + 2)?;
        let len = u32le(d, e + 4)? as usize;
        let data_off = u32le(d, e + 12)? as usize;
        let ks = key_tab + key_off;
        let kend = ks + d.get(ks..)?.iter().position(|&b| b == 0)?;
        let key = String::from_utf8_lossy(&d[ks..kend]).into_owned();
        let v = d.get(data_tab + data_off..data_tab + data_off + len)?;
        let val = match fmt {
            0x0404 => u32le(v, 0).map(|x| x.to_string()).unwrap_or_default(),
            _ => String::from_utf8_lossy(v)
                .trim_end_matches('\0')
                .to_string(),
        };
        out.insert(key, val);
    }
    Some(out)
}

#[derive(Debug, PartialEq)]
pub enum Kind {
    /// Plain 64-bit big-endian PowerPC ELF: what the pipeline wants.
    Elf,
    /// Retail SELF: encrypted, needs RPCS3 (or another decrypter) first.
    RetailSelf,
    /// Debug/"fake" SELF: unwrappable without keys (ps3recomp/tools/unfself.py).
    DebugSelf,
    Unknown,
}

pub fn classify(path: &Path) -> Result<Kind> {
    let mut f = fs::File::open(path).ctx(|| format!("opening {}", path.display()))?;
    let mut h = [0u8; 0x40];
    use std::io::Read;
    let n = f.read(&mut h)?;
    let h = &h[..n];
    if h.starts_with(b"\x7fELF") {
        // ELFCLASS64, big-endian, e_machine 21 (PPC64)
        if h.len() >= 0x14 && h[4] == 2 && h[5] == 2 && u16be(h, 0x12) == Some(21) {
            return Ok(Kind::Elf);
        }
        return Ok(Kind::Unknown);
    }
    if h.starts_with(b"SCE\0") {
        return Ok(if u16be(h, 8) == Some(0x8000) {
            Kind::DebugSelf
        } else {
            Kind::RetailSelf
        });
    }
    Ok(Kind::Unknown)
}

/// RPCS3's command line decrypter ("--decrypt <path>") writes EBOOT.elf next to EBOOT.BIN.
pub fn decrypt_with_rpcs3(rpcs3: &Path, bin: &Path) -> Result<()> {
    util::info(&format!("Running RPCS3 to decrypt {} ...", bin.display()));
    let mut cmd = Command::new(rpcs3);
    cmd.arg("--decrypt").arg(bin);
    if let Some(d) = rpcs3.parent() {
        cmd.current_dir(d);
    }
    util::run(&mut cmd)
}

pub fn manual_decrypt_help(bin: &Path) -> String {
    format!(
        "Your EBOOT.BIN is encrypted (a retail SELF). The installer needs the decrypted EBOOT.elf.\n\
         \n\
         Easiest: re-run with  --rpcs3 \"C:\\path\\to\\rpcs3.exe\"  and the installer will ask RPCS3 to do it.\n\
         Or by hand in RPCS3:  Utilities -> Decrypt PS3 Binaries -> pick\n    {}\n\
         RPCS3 writes EBOOT.elf next to it; then run the installer again.",
        bin.display()
    )
}

/// WWS job modules embedded in the PPU ELF. Layout (big-endian words):
///   +0x00 four `ila` instructions | +0x10 entry offset | +0x14 image size | +0x18 | +0x1C | +0x20 0xC0DEC0DE
/// Saved as jobmod_<sha1[:12]>_e<entry, %04X>.bin, which is how the port has always named them.
pub fn extract_jobmods(elf: &Path, out_dir: &Path) -> Result<Vec<String>> {
    let d = fs::read(elf).ctx(|| format!("reading {}", elf.display()))?;
    fs::create_dir_all(out_dir).ctx(|| format!("creating {}", out_dir.display()))?;
    let mut names = Vec::new();
    let magic = [0xC0, 0xDE, 0xC0, 0xDE];
    let mut i = 0x20;
    while i + 4 <= d.len() {
        if d[i..i + 4] == magic && (i - 0x20) % 4 == 0 {
            let start = i - 0x20;
            let entry = u32be(&d, start + 0x10).unwrap_or(0);
            let size = u32be(&d, start + 0x14).unwrap_or(0) as usize;
            // Sanity: the module begins with four `ila` (opcode 0b0100001) and fits in local store.
            let ila = (0..4).all(|k| {
                u32be(&d, start + 4 * k)
                    .map(|w| w >> 25 == 0x21)
                    .unwrap_or(false)
            });
            if ila
                && (0x24..=0x40000).contains(&size)
                && start + size <= d.len()
                && (entry as usize) < size
            {
                let m = &d[start..start + size];
                let name = format!("jobmod_{}_e{:04X}", &hash::hex(&hash::sha1(m))[..12], entry);
                let path = out_dir.join(format!("{name}.bin"));
                if !names.contains(&name) {
                    fs::write(&path, m).ctx(|| format!("writing {}", path.display()))?;
                    names.push(name);
                }
                i = start + size.max(0x24);
                continue;
            }
        }
        i += 4;
    }
    if names.is_empty() {
        bail!("no WWS job modules (C0DEC0DE) found in {}", elf.display());
    }
    Ok(names)
}

/// Copy `len` bytes at `offset` of `src` into `dst`.
pub fn carve(src: &Path, offset: u64, len: u64, dst: &Path) -> Result<()> {
    let d = fs::read(src).ctx(|| format!("reading {}", src.display()))?;
    let (o, l) = (offset as usize, len as usize);
    if o.checked_add(l).map(|e| e > d.len()).unwrap_or(true) {
        bail!(
            "carve {:#x}+{:#x} is past the end of {} ({} bytes)",
            offset,
            len,
            src.display(),
            d.len()
        );
    }
    util::write_file(dst, &d[o..o + l])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sfo_roundtrip() {
        // Build a 2-entry SFO by hand.
        let keys = b"TITLE\0TITLE_ID\0\0";
        let title = b"inFamous\0\0\0\0";
        let tid = b"BCES00220\0\0\0\0\0\0\0";
        let n = 2u32;
        let key_tab = 20 + 16 * n;
        let data_tab = key_tab + keys.len() as u32;
        let mut d = Vec::new();
        d.extend_from_slice(b"\0PSF");
        d.extend_from_slice(&0x0101u32.to_le_bytes());
        d.extend_from_slice(&key_tab.to_le_bytes());
        d.extend_from_slice(&data_tab.to_le_bytes());
        d.extend_from_slice(&n.to_le_bytes());
        for (ko, len, max, off) in [(0u16, 9u32, 12u32, 0u32), (6, 10, 16, 12)] {
            d.extend_from_slice(&ko.to_le_bytes());
            d.extend_from_slice(&0x0204u16.to_le_bytes());
            d.extend_from_slice(&len.to_le_bytes());
            d.extend_from_slice(&max.to_le_bytes());
            d.extend_from_slice(&off.to_le_bytes());
        }
        d.extend_from_slice(keys);
        d.extend_from_slice(title);
        d.extend_from_slice(tid);
        let m = parse_sfo(&d).unwrap();
        assert_eq!(m["TITLE"], "inFamous");
        assert_eq!(m["TITLE_ID"], "BCES00220");
    }
}
