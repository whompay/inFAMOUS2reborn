//! Reading a PS3 disc image (.iso): ISO 9660 (Joliet names when present, multi-extent files)
//! plus the PS3 disc encryption used by 1:1 dumps (Redump).
//!
//! PS3 disc encryption: sector 0 lists the *plain* regions as (first, last) sector pairs
//! (big-endian u32 count at 0, pairs from offset 8). Every other sector is AES-128-CBC encrypted
//! with the disc key, IV = the sector number as a big-endian 128-bit integer. Redump publishes the
//! disc key as a `.dkey` file (32 hex digits). "Decrypted" ISOs keep the same sector 0 but store
//! everything in the clear; `detect_encryption` tells the two apart from file signatures.

use crate::bail;
use crate::util::{self, Ctx, Result};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const SECTOR: u64 = 2048;

// ---------------------------------------------------------------------------------------------
// AES-128 decryption (table-driven; tables generated at first use)
// ---------------------------------------------------------------------------------------------

struct Tables {
    sbox: [u8; 256],
    inv: [u8; 256],
    td: [[u32; 256]; 4],
}

fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    while b != 0 {
        if b & 1 != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1B;
        }
        b >>= 1;
    }
    p
}

fn tables() -> &'static Tables {
    static T: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut sbox = [0u8; 256];
        let mut inv = [0u8; 256];
        for x in 0..256usize {
            // multiplicative inverse in GF(2^8), then the affine map
            let b = if x == 0 {
                0
            } else {
                (1..=255u8).find(|&y| gmul(x as u8, y) == 1).unwrap()
            };
            let s = b
                ^ b.rotate_left(1)
                ^ b.rotate_left(2)
                ^ b.rotate_left(3)
                ^ b.rotate_left(4)
                ^ 0x63;
            sbox[x] = s;
            inv[s as usize] = x as u8;
        }
        let mut td = [[0u32; 256]; 4];
        for x in 0..256 {
            let s = inv[x];
            let w = (gmul(s, 0x0e) as u32) << 24
                | (gmul(s, 0x09) as u32) << 16
                | (gmul(s, 0x0d) as u32) << 8
                | gmul(s, 0x0b) as u32;
            td[0][x] = w;
            td[1][x] = w.rotate_right(8);
            td[2][x] = w.rotate_right(16);
            td[3][x] = w.rotate_right(24);
        }
        Tables { sbox, inv, td }
    })
}

#[derive(Clone)]
pub struct Aes128Dec {
    rk: [u32; 44],
}

impl Aes128Dec {
    pub fn new(key: &[u8; 16]) -> Self {
        let t = tables();
        let mut w = [0u32; 44];
        for i in 0..4 {
            w[i] = u32::from_be_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
        }
        let mut rcon = 1u8;
        for i in 4..44 {
            let mut tmp = w[i - 1];
            if i % 4 == 0 {
                let r = tmp.rotate_left(8);
                tmp = (t.sbox[(r >> 24) as usize] as u32) << 24
                    | (t.sbox[((r >> 16) & 0xff) as usize] as u32) << 16
                    | (t.sbox[((r >> 8) & 0xff) as usize] as u32) << 8
                    | t.sbox[(r & 0xff) as usize] as u32;
                tmp ^= (rcon as u32) << 24;
                rcon = gmul(rcon, 2);
            }
            w[i] = w[i - 4] ^ tmp;
        }
        // Equivalent inverse cipher: reverse the round keys, InvMixColumns on rounds 1..9.
        let mut rk = [0u32; 44];
        for r in 0..11 {
            for c in 0..4 {
                rk[4 * r + c] = w[4 * (10 - r) + c];
            }
        }
        for v in rk.iter_mut().take(40).skip(4) {
            let x = *v;
            *v = t.td[0][t.sbox[(x >> 24) as usize] as usize]
                ^ t.td[1][t.sbox[((x >> 16) & 0xff) as usize] as usize]
                ^ t.td[2][t.sbox[((x >> 8) & 0xff) as usize] as usize]
                ^ t.td[3][t.sbox[(x & 0xff) as usize] as usize];
        }
        Aes128Dec { rk }
    }

    pub fn decrypt_block(&self, b: &mut [u8]) {
        let t = tables();
        let td = &t.td;
        let rk = &self.rk;
        let ld =
            |i: usize| u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        let (mut s0, mut s1, mut s2, mut s3) =
            (ld(0) ^ rk[0], ld(1) ^ rk[1], ld(2) ^ rk[2], ld(3) ^ rk[3]);
        let f = |a: u32, b: u32, c: u32, d: u32, k: u32| {
            td[0][(a >> 24) as usize]
                ^ td[1][((b >> 16) & 0xff) as usize]
                ^ td[2][((c >> 8) & 0xff) as usize]
                ^ td[3][(d & 0xff) as usize]
                ^ k
        };
        for r in 1..10 {
            let k = &rk[4 * r..4 * r + 4];
            let t0 = f(s0, s3, s2, s1, k[0]);
            let t1 = f(s1, s0, s3, s2, k[1]);
            let t2 = f(s2, s1, s0, s3, k[2]);
            let t3 = f(s3, s2, s1, s0, k[3]);
            (s0, s1, s2, s3) = (t0, t1, t2, t3);
        }
        let si = &t.inv;
        let g = |a: u32, b: u32, c: u32, d: u32, k: u32| {
            ((si[(a >> 24) as usize] as u32) << 24
                | (si[((b >> 16) & 0xff) as usize] as u32) << 16
                | (si[((c >> 8) & 0xff) as usize] as u32) << 8
                | si[(d & 0xff) as usize] as u32)
                ^ k
        };
        let o = [
            g(s0, s3, s2, s1, rk[40]),
            g(s1, s0, s3, s2, rk[41]),
            g(s2, s1, s0, s3, rk[42]),
            g(s3, s2, s1, s0, rk[43]),
        ];
        for i in 0..4 {
            b[4 * i..4 * i + 4].copy_from_slice(&o[i].to_be_bytes());
        }
    }

    /// AES-128-CBC decrypt in place.
    pub fn decrypt_cbc(&self, iv: &[u8; 16], data: &mut [u8]) {
        let mut prev = *iv;
        for blk in data.chunks_exact_mut(16) {
            let mut ct = [0u8; 16];
            ct.copy_from_slice(blk);
            self.decrypt_block(blk);
            for i in 0..16 {
                blk[i] ^= prev[i];
            }
            prev = ct;
        }
    }
}

/// Parse a disc key: a `.dkey` file's text (32 hex digits) or 16 raw bytes.
pub fn parse_key(data: &[u8]) -> Result<[u8; 16]> {
    if data.len() == 16 && !data.iter().all(|b| b.is_ascii_hexdigit()) {
        let mut k = [0u8; 16];
        k.copy_from_slice(data);
        return Ok(k);
    }
    let s: String = String::from_utf8_lossy(data)
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    if s.len() != 32 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("a disc key is 32 hexadecimal digits");
    }
    let mut k = [0u8; 16];
    for i in 0..16 {
        k[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
    }
    Ok(k)
}

pub fn read_key_file(p: &Path) -> Result<[u8; 16]> {
    let d = fs::read(p).ctx(|| format!("reading {}", p.display()))?;
    parse_key(&d).ctx(|| format!("{} is not a disc key", p.display()))
}

/// `<iso stem>.dkey` (or `.key`) next to the image, if there is one.
pub fn find_key_file(iso: &Path) -> Option<PathBuf> {
    let dir = iso.parent()?;
    let stem = iso.file_stem()?.to_string_lossy().to_string();
    for ext in ["dkey", "key", "DKEY", "KEY"] {
        let p = dir.join(format!("{stem}.{ext}"));
        if p.is_file() {
            return Some(p);
        }
    }
    // A keys folder next to the image.
    for sub in ["keys", "dkeys", "disc_keys"] {
        let p = dir.join(sub).join(format!("{stem}.dkey"));
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------
// The image
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Entry {
    /// Forward-slash path from the disc root, e.g. `PS3_GAME/USRDIR/EBOOT.BIN`.
    pub path: String,
    /// (first sector, byte length) of each extent, in order.
    pub extents: Vec<(u32, u64)>,
    pub size: u64,
}

pub struct Iso {
    f: File,
    pub sectors: u64,
    /// Inclusive plain sector ranges from sector 0 (empty: not a PS3 disc layout).
    plain: Vec<(u32, u32)>,
    aes: Option<Aes128Dec>,
    /// Apply decryption to sectors outside the plain regions.
    pub decrypting: bool,
    pub files: Vec<Entry>,
}

fn u32le(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}
fn u32be(d: &[u8], o: usize) -> u32 {
    u32::from_be_bytes(d[o..o + 4].try_into().unwrap())
}

/// The first bytes a file of this name always starts with.
fn signature(path: &str) -> Option<&'static [u8]> {
    let l = path.to_ascii_lowercase();
    Some(if l.ends_with(".sfo") {
        b"\0PSF"
    } else if l.ends_with("eboot.bin") || l.ends_with(".self") || l.ends_with(".sprx") {
        b"SCE\0"
    } else if l.ends_with(".png") {
        b"\x89PNG"
    } else if l.contains(".psarc") {
        b"PSAR"
    } else if l.ends_with(".elf") || l.ends_with(".prx") {
        b"\x7fELF"
    } else {
        return None;
    })
}

impl Iso {
    pub fn open(path: &Path) -> Result<Iso> {
        let mut f = File::open(path).ctx(|| format!("opening {}", path.display()))?;
        let len = f.metadata()?.len();
        if len < 18 * SECTOR {
            bail!("{} is too small to be a disc image", path.display());
        }
        let mut s0 = vec![0u8; SECTOR as usize];
        f.read_exact(&mut s0)?;
        let sectors = len / SECTOR;
        let mut plain = Vec::new();
        let n = u32be(&s0, 0) as usize;
        if (1..=200).contains(&n) && 8 + 8 * n <= s0.len() {
            for k in 0..n {
                let (a, b) = (u32be(&s0, 8 + 8 * k), u32be(&s0, 12 + 8 * k));
                if a > b || b as u64 >= sectors.max(1) + 1 {
                    plain.clear();
                    break;
                }
                plain.push((a, b));
            }
            // Sanity: sorted, starting at sector 0.
            if plain.first().map(|r| r.0) != Some(0) || plain.windows(2).any(|w| w[1].0 <= w[0].1) {
                plain.clear();
            }
        }
        let mut iso = Iso {
            f,
            sectors,
            plain,
            aes: None,
            decrypting: false,
            files: Vec::new(),
        };
        iso.files = iso
            .list()
            .ctx(|| format!("{} is not a readable ISO 9660 disc image", path.display()))?;
        Ok(iso)
    }

    /// True if this looks like a PS3 disc layout with encrypted regions.
    pub fn has_regions(&self) -> bool {
        !self.plain.is_empty()
            && !(self.plain.len() == 1 && self.plain[0].1 as u64 + 1 >= self.sectors)
    }

    pub fn is_plain_sector(&self, s: u64) -> bool {
        self.plain.is_empty()
            || self
                .plain
                .iter()
                .any(|&(a, b)| s >= a as u64 && s <= b as u64)
    }

    pub fn set_key(&mut self, key: &[u8; 16]) {
        self.aes = Some(Aes128Dec::new(key));
    }

    pub fn has_key(&self) -> bool {
        self.aes.is_some()
    }

    /// Read `count` sectors from `first`, decrypting where needed.
    pub fn read_sectors(&mut self, first: u64, count: u64, buf: &mut Vec<u8>) -> Result<()> {
        buf.resize((count * SECTOR) as usize, 0);
        self.f.seek(SeekFrom::Start(first * SECTOR))?;
        self.f.read_exact(buf).ctx(|| {
            format!(
                "reading sectors {first}..{} (the image may be truncated)",
                first + count
            )
        })?;
        if self.decrypting {
            let aes = self
                .aes
                .clone()
                .ctx(|| "this disc image is encrypted and needs its disc key")?;
            for i in 0..count {
                let s = first + i;
                if self.is_plain_sector(s) {
                    continue;
                }
                let mut iv = [0u8; 16];
                iv[8..].copy_from_slice(&s.to_be_bytes());
                let off = (i * SECTOR) as usize;
                aes.decrypt_cbc(&iv, &mut buf[off..off + SECTOR as usize]);
            }
        }
        Ok(())
    }

    fn list(&mut self) -> Result<Vec<Entry>> {
        let mut buf = Vec::new();
        // Volume descriptors from sector 16: prefer Joliet (type 2 with a UCS-2 escape).
        let mut root: Option<(u32, u32, bool)> = None;
        for s in 16..64u64 {
            self.read_sectors(s, 1, &mut buf)?;
            if &buf[1..6] != b"CD001" {
                bail!("no ISO 9660 volume descriptor at sector {s}");
            }
            let t = buf[0];
            if t == 255 {
                break;
            }
            let rec = &buf[156..190];
            let (lba, len) = (u32le(rec, 2), u32le(rec, 10));
            if t == 1 && root.is_none() {
                root = Some((lba, len, false));
            }
            if t == 2 && buf[88] == b'%' && buf[89] == b'/' && matches!(buf[90], b'@' | b'C' | b'E')
            {
                root = Some((lba, len, true));
            }
        }
        let (lba, len, joliet) = root.ctx(|| "no primary volume descriptor")?;
        let mut out = Vec::new();
        let mut stack = vec![(String::new(), lba, len, 0usize)];
        while let Some((prefix, lba, len, depth)) = stack.pop() {
            if depth > 32 {
                bail!("directory tree too deep");
            }
            let nsec = (len as u64).div_ceil(SECTOR);
            self.read_sectors(lba as u64, nsec, &mut buf)?;
            let data = buf.clone();
            let mut pos = 0usize;
            let mut pending: Option<Entry> = None;
            while pos < data.len() {
                let rl = data[pos] as usize;
                if rl == 0 {
                    pos = (pos / SECTOR as usize + 1) * SECTOR as usize; // records don't cross sectors
                    continue;
                }
                if pos + rl > data.len() || rl < 34 {
                    break;
                }
                let r = &data[pos..pos + rl];
                pos += rl;
                let nl = r[32] as usize;
                if 33 + nl > r.len() {
                    continue;
                }
                let raw = &r[33..33 + nl];
                if nl == 1 && (raw[0] == 0 || raw[0] == 1) {
                    continue; // . and ..
                }
                let mut name = if joliet {
                    let u: Vec<u16> = raw
                        .chunks_exact(2)
                        .map(|c| u16::from_be_bytes([c[0], c[1]]))
                        .collect();
                    String::from_utf16_lossy(&u)
                } else {
                    String::from_utf8_lossy(raw).into_owned()
                };
                if let Some(i) = name.find(';') {
                    name.truncate(i);
                }
                if name.ends_with('.') {
                    name.pop();
                }
                if name.is_empty() || name.contains(['/', '\\']) || name == ".." {
                    continue;
                }
                let (elba, elen, flags) = (u32le(r, 2), u32le(r, 10), r[25]);
                let full = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}/{name}")
                };
                if flags & 2 != 0 {
                    stack.push((full, elba, elen, depth + 1));
                    continue;
                }
                // Multi-extent files: records with flag 0x80 continue in the next record.
                let e = match pending.take() {
                    Some(mut p) if p.path == full => {
                        p.extents.push((elba, elen as u64));
                        p.size += elen as u64;
                        p
                    }
                    Some(p) => {
                        out.push(p);
                        Entry {
                            path: full,
                            extents: vec![(elba, elen as u64)],
                            size: elen as u64,
                        }
                    }
                    None => Entry {
                        path: full,
                        extents: vec![(elba, elen as u64)],
                        size: elen as u64,
                    },
                };
                if flags & 0x80 != 0 {
                    pending = Some(e);
                } else {
                    out.push(e);
                }
            }
            if let Some(p) = pending {
                out.push(p);
            }
        }
        out.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(out)
    }

    pub fn find(&self, path: &str) -> Option<Entry> {
        self.files
            .iter()
            .find(|e| e.path.eq_ignore_ascii_case(path))
            .cloned()
    }

    /// The first `max` bytes of a file.
    pub fn read_head(&mut self, e: &Entry, max: u64) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let mut buf = Vec::new();
        for &(lba, len) in &e.extents {
            let want = len.min(max - out.len() as u64);
            if want == 0 {
                break;
            }
            self.read_sectors(lba as u64, want.div_ceil(SECTOR), &mut buf)?;
            out.extend_from_slice(&buf[..want as usize]);
        }
        Ok(out)
    }

    pub fn read_file(&mut self, e: &Entry) -> Result<Vec<u8>> {
        self.read_head(e, e.size)
    }

    /// Decide whether the encrypted regions really are encrypted, from files that start in one
    /// and have a known signature. Some(true) = encrypted, Some(false) = stored in the clear,
    /// None = could not tell (no such file).
    pub fn detect_encryption(&mut self) -> Result<Option<bool>> {
        if !self.has_regions() {
            return Ok(Some(false));
        }
        let saved = self.decrypting;
        self.decrypting = false;
        let mut verdict = None;
        let files = self.files.clone();
        let mut buf = Vec::new();
        for e in files.iter().filter(|e| e.size >= 4) {
            let lba = e.extents[0].0 as u64;
            if self.is_plain_sector(lba) {
                continue;
            }
            let Some(expect) = signature(&e.path) else {
                continue;
            };
            self.read_sectors(lba, 1, &mut buf)?;
            verdict = Some(!buf.starts_with(expect));
            break;
        }
        self.decrypting = saved;
        Ok(verdict)
    }

    /// Check a key against a file in an encrypted region with a known signature.
    pub fn key_works(&mut self) -> Result<bool> {
        let saved = self.decrypting;
        self.decrypting = true;
        let files = self.files.clone();
        let mut res = None;
        let mut buf = Vec::new();
        for e in files.iter().filter(|e| e.size >= 4) {
            let lba = e.extents[0].0 as u64;
            if self.is_plain_sector(lba) {
                continue;
            }
            let Some(m) = signature(&e.path) else {
                continue;
            };
            let r = self.read_sectors(lba, 1, &mut buf);
            res = Some(r.is_ok() && buf.starts_with(m));
            break;
        }
        self.decrypting = saved;
        Ok(res.unwrap_or(true))
    }

    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|e| e.size).sum()
    }

    /// Copy every file to `out` (created), calling `progress(done_bytes, total_bytes)` as it goes.
    pub fn extract(&mut self, out: &Path, progress: &mut dyn FnMut(u64, u64, &str)) -> Result<()> {
        let total = self.total_size();
        let mut done = 0u64;
        let files = self.files.clone();
        let mut buf = Vec::new();
        const CHUNK: u64 = 512; // sectors (1 MiB)
        for e in &files {
            let dst = util::join_slash(out, &e.path);
            util::ensure_parent(&dst)?;
            let tmp = dst.with_extension("partial");
            let mut w = std::io::BufWriter::with_capacity(
                1 << 20,
                File::create(&tmp).ctx(|| format!("creating {}", tmp.display()))?,
            );
            for &(lba, len) in &e.extents {
                let mut left = len;
                let mut s = lba as u64;
                while left > 0 {
                    let n = left.div_ceil(SECTOR).min(CHUNK);
                    self.read_sectors(s, n, &mut buf)?;
                    let take = left.min(n * SECTOR) as usize;
                    w.write_all(&buf[..take])
                        .ctx(|| format!("writing {}", dst.display()))?;
                    left -= take as u64;
                    s += n;
                    done += take as u64;
                    progress(done, total, &e.path);
                }
            }
            w.flush()?;
            drop(w);
            if dst.exists() {
                fs::remove_file(&dst)?;
            }
            fs::rename(&tmp, &dst)?;
        }
        Ok(())
    }
}

/// Open an image and get it ready to read: detect encryption, apply the key (given, or a `.dkey`
/// next to the image) and verify it.
pub fn open_ready(path: &Path, key: Option<[u8; 16]>) -> Result<Iso> {
    let mut iso = Iso::open(path)?;
    let key = match key {
        Some(k) => Some(k),
        None => match find_key_file(path) {
            Some(p) => Some(read_key_file(&p)?),
            None => None,
        },
    };
    match iso.detect_encryption()? {
        Some(false) => {}
        enc => {
            let Some(k) = key else {
                if enc == Some(true) {
                    bail!("this disc image is encrypted (a 1:1 dump); it needs its disc key (.dkey file)");
                }
                return Ok(iso);
            };
            iso.set_key(&k);
            if !iso.key_works()? {
                bail!("the disc key does not fit this disc image");
            }
            iso.decrypting = true;
        }
    }
    Ok(iso)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aes_fips197() {
        // FIPS-197 C.1
        let key: [u8; 16] = core::array::from_fn(|i| i as u8);
        let mut ct = [
            0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4,
            0xc5, 0x5a,
        ];
        Aes128Dec::new(&key).decrypt_block(&mut ct);
        let pt: [u8; 16] = core::array::from_fn(|i| (i as u8) * 0x11);
        assert_eq!(ct, pt);
        // FIPS-197 / SP800-38A F.2.2 CBC-AES128.Decrypt first block
        let key = parse_key(b"2b7e151628aed2a6abf7158809cf4f3c").unwrap();
        let iv: [u8; 16] = core::array::from_fn(|i| i as u8);
        let mut d = [
            0x76, 0x49, 0xab, 0xac, 0x81, 0x19, 0xb2, 0x46, 0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9,
            0x19, 0x7d,
        ];
        Aes128Dec::new(&key).decrypt_cbc(&iv, &mut d);
        assert_eq!(
            d,
            [
                0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93,
                0x17, 0x2a
            ]
        );
    }

    #[test]
    fn keys() {
        assert!(parse_key(b"00112233445566778899aabbccddeeff\r\n").is_ok());
        assert!(parse_key(b"0011").is_err());
    }
}
