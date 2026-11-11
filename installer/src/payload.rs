//! The files a setup .exe carries: appended to the executable after it is built.
//!
//!   <exe> <entries...> <u64 LE entries length> <u64 LE file count> "IFPAYLD1"
//!   entry = <u32 LE path length> <path, utf-8, forward slashes> <u64 LE data length> <data>

use crate::bail;
use crate::util::{self, Ctx, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"IFPAYLD1";

/// Write `stub` + the files (`(published path, source file)`) to `out`.
pub fn write(stub: &Path, files: &[(String, PathBuf)], out: &Path) -> Result<u64> {
    util::ensure_parent(out)?;
    let mut w = BufWriter::with_capacity(
        1 << 20,
        File::create(out).ctx(|| format!("creating {}", out.display()))?,
    );
    let exe = std::fs::read(stub).ctx(|| format!("reading {}", stub.display()))?;
    if find(&exe).is_some() {
        bail!(
            "{} already carries a payload; use the freshly built infamous2-setup.exe",
            stub.display()
        );
    }
    w.write_all(&exe)?;
    let mut len = 0u64;
    for (rel, src) in files {
        let mut f = File::open(src).ctx(|| format!("opening {}", src.display()))?;
        let size = f.metadata()?.len();
        w.write_all(&(rel.len() as u32).to_le_bytes())?;
        w.write_all(rel.as_bytes())?;
        w.write_all(&size.to_le_bytes())?;
        let n = std::io::copy(&mut f, &mut w)?;
        if n != size {
            bail!("{} changed while it was being packed", src.display());
        }
        len += 4 + rel.len() as u64 + 8 + size;
    }
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&(files.len() as u64).to_le_bytes())?;
    w.write_all(MAGIC)?;
    w.flush()?;
    Ok(exe.len() as u64 + len + 24)
}

/// (offset of the entries, entries length, file count) if `data` ends with a payload.
fn find(data: &[u8]) -> Option<(u64, u64, u64)> {
    if data.len() < 24 || &data[data.len() - 8..] != MAGIC {
        return None;
    }
    let t = data.len() - 24;
    let len = u64::from_le_bytes(data[t..t + 8].try_into().ok()?);
    let count = u64::from_le_bytes(data[t + 8..t + 16].try_into().ok()?);
    let start = (t as u64).checked_sub(len)?;
    Some((start, len, count))
}

pub struct Payload {
    exe: PathBuf,
    start: u64,
    len: u64,
    pub count: u64,
}

impl Payload {
    /// The payload of an executable (normally `std::env::current_exe()`), if it has one.
    pub fn open(exe: &Path) -> Result<Option<Payload>> {
        let mut f = File::open(exe).ctx(|| format!("opening {}", exe.display()))?;
        let size = f.metadata()?.len();
        if size < 24 {
            return Ok(None);
        }
        let mut tail = [0u8; 24];
        f.seek(SeekFrom::Start(size - 24))?;
        f.read_exact(&mut tail)?;
        if &tail[16..] != MAGIC {
            return Ok(None);
        }
        let len = u64::from_le_bytes(tail[0..8].try_into().unwrap());
        let count = u64::from_le_bytes(tail[8..16].try_into().unwrap());
        let start = (size - 24).checked_sub(len).ctx(|| "damaged payload")?;
        Ok(Some(Payload {
            exe: exe.to_path_buf(),
            start,
            len,
            count,
        }))
    }

    /// One file's contents, by its forward-slash path.
    pub fn read(&self, want: &str) -> Result<Option<Vec<u8>>> {
        let mut f = File::open(&self.exe)?;
        let mut pos = self.start;
        let end = self.start + self.len;
        while pos < end {
            f.seek(SeekFrom::Start(pos))?;
            let mut l4 = [0u8; 4];
            f.read_exact(&mut l4)?;
            let pl = u32::from_le_bytes(l4) as u64;
            if pl == 0 || pl > 4096 {
                bail!("damaged payload");
            }
            let mut pb = vec![0u8; pl as usize];
            f.read_exact(&mut pb)?;
            let mut l8 = [0u8; 8];
            f.read_exact(&mut l8)?;
            let size = u64::from_le_bytes(l8);
            if pb == want.as_bytes() {
                let mut d = vec![0u8; size as usize];
                f.read_exact(&mut d)?;
                return Ok(Some(d));
            }
            pos += 4 + pl + 8 + size;
        }
        Ok(None)
    }

    pub fn bytes(&self) -> u64 {
        self.len
    }

    /// Every file's forward-slash path, in payload order.
    pub fn names(&self) -> Result<Vec<String>> {
        let mut f = File::open(&self.exe)?;
        let mut pos = self.start;
        let end = self.start + self.len;
        let mut out = Vec::new();
        while pos < end {
            f.seek(SeekFrom::Start(pos))?;
            let mut l4 = [0u8; 4];
            f.read_exact(&mut l4)?;
            let pl = u32::from_le_bytes(l4) as u64;
            if pl == 0 || pl > 4096 {
                bail!("damaged payload");
            }
            let mut pb = vec![0u8; pl as usize];
            f.read_exact(&mut pb)?;
            let mut l8 = [0u8; 8];
            f.read_exact(&mut l8)?;
            let size = u64::from_le_bytes(l8);
            out.push(String::from_utf8_lossy(&pb).into_owned());
            pos += 4 + pl + 8 + size;
        }
        Ok(out)
    }

    /// Write every file under `dest`; `progress(done_bytes, total_bytes)`.
    pub fn extract(&self, dest: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<usize> {
        let mut f = File::open(&self.exe)?;
        f.seek(SeekFrom::Start(self.start))?;
        let mut r = BufReader::with_capacity(1 << 20, f.take(self.len));
        let mut done = 0u64;
        let mut n = 0;
        loop {
            let mut l4 = [0u8; 4];
            match r.read_exact(&mut l4) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let pl = u32::from_le_bytes(l4) as usize;
            if pl == 0 || pl > 4096 {
                bail!("damaged payload (path length {pl})");
            }
            let mut pb = vec![0u8; pl];
            r.read_exact(&mut pb)?;
            let rel =
                String::from_utf8(pb).map_err(|_| util::Error("damaged payload (path)".into()))?;
            if rel.split('/').any(|s| {
                s.is_empty() || s == ".." || s == "." || s.contains(':') || s.contains('\\')
            }) {
                bail!("damaged payload (bad path {rel})");
            }
            let mut l8 = [0u8; 8];
            r.read_exact(&mut l8)?;
            let size = u64::from_le_bytes(l8);
            let dst = util::join_slash(dest, &rel);
            util::ensure_parent(&dst)?;
            let tmp = dst.with_extension("setup-tmp");
            {
                let mut w = BufWriter::new(
                    File::create(&tmp).ctx(|| format!("creating {}", tmp.display()))?,
                );
                let copied = std::io::copy(&mut (&mut r).take(size), &mut w)?;
                if copied != size {
                    bail!("damaged payload ({rel} is truncated)");
                }
                w.flush()?;
            }
            if dst.exists() {
                std::fs::remove_file(&dst).ctx(|| {
                    format!(
                        "replacing {} (is it open in another program?)",
                        dst.display()
                    )
                })?;
            }
            std::fs::rename(&tmp, &dst)?;
            done += 4 + pl as u64 + 8 + size;
            n += 1;
            progress(done, self.len);
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let d = std::env::temp_dir().join(format!("payload-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("src/sub")).unwrap();
        std::fs::write(d.join("stub.exe"), b"MZ stub").unwrap();
        std::fs::write(d.join("src/a.txt"), b"hello").unwrap();
        std::fs::write(d.join("src/sub/b.bin"), vec![7u8; 100_000]).unwrap();
        std::fs::write(d.join("src/empty"), b"").unwrap();
        let files = vec![
            ("a.txt".to_string(), d.join("src/a.txt")),
            ("x/y/b.bin".to_string(), d.join("src/sub/b.bin")),
            ("empty".to_string(), d.join("src/empty")),
        ];
        write(&d.join("stub.exe"), &files, &d.join("setup.exe")).unwrap();
        let p = Payload::open(&d.join("setup.exe")).unwrap().unwrap();
        assert_eq!(p.count, 3);
        let n = p.extract(&d.join("out"), &mut |_, _| {}).unwrap();
        assert_eq!(n, 3);
        assert_eq!(std::fs::read(d.join("out/a.txt")).unwrap(), b"hello");
        assert_eq!(
            std::fs::read(d.join("out/x/y/b.bin")).unwrap().len(),
            100_000
        );
        assert!(Payload::open(&d.join("stub.exe")).unwrap().is_none());
        assert_eq!(p.read("x/y/b.bin").unwrap().unwrap().len(), 100_000);
        assert_eq!(p.read("a.txt").unwrap().unwrap(), b"hello");
        assert!(p.read("nope").unwrap().is_none());
        assert_eq!(p.names().unwrap(), vec!["a.txt", "x/y/b.bin", "empty"]);
        let _ = std::fs::remove_dir_all(&d);
    }
}
