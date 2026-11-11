//! `installer/generated.lock`: what the maintainer's pipeline produced from their EBOOT, and what
//! each generated file must look like after patching. Written by `make-patches`, read by `install`.
//!
//!   eboot <sha256 of EBOOT.elf>
//!   file <pipeline sha256 | -> <final sha256 | -> <path>
//!
//! pipeline `-` = the file is not produced by the pipeline (its patch creates it from nothing);
//! final `-`    = the pipeline produces it but the port does not use it (it is deleted).
//! Each file whose two hashes differ has `installer/patches/<path>.patch`.

use crate::bail;
use crate::util::{Ctx, Result};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub path: String,
    pub base: Option<String>,
    pub fin: Option<String>,
}

#[derive(Debug, Default)]
pub struct Lock {
    pub eboot: Option<String>,
    pub files: Vec<Entry>,
}

fn opt(s: &str) -> Option<String> {
    if s == "-" {
        None
    } else {
        Some(s.to_string())
    }
}

impl Lock {
    pub fn load(path: &Path) -> Result<Option<Lock>> {
        if !path.exists() {
            return Ok(None);
        }
        let txt = std::fs::read_to_string(path).ctx(|| format!("reading {}", path.display()))?;
        let mut lock = Lock::default();
        for (n, line) in txt.lines().enumerate() {
            let line = line.trim_end();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("eboot ") {
                lock.eboot = Some(rest.trim().to_ascii_lowercase());
            } else if let Some(rest) = line.strip_prefix("file ") {
                let mut it = rest.splitn(3, ' ');
                let (Some(b), Some(f), Some(p)) = (it.next(), it.next(), it.next()) else {
                    bail!(
                        "{} line {}: expected `file <base> <final> <path>`",
                        path.display(),
                        n + 1
                    );
                };
                lock.files.push(Entry {
                    path: p.to_string(),
                    base: opt(b),
                    fin: opt(f),
                });
            } else {
                bail!("{} line {}: unknown entry", path.display(), n + 1);
            }
        }
        Ok(Some(lock))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut s = String::new();
        s.push_str("# Written by `infamous2-installer make-patches`. Do not edit by hand.\n");
        s.push_str("# file <hash the pipeline produces> <hash after the port's patches> <path>\n");
        if let Some(e) = &self.eboot {
            s.push_str(&format!("eboot {e}\n"));
        }
        for f in &self.files {
            s.push_str(&format!(
                "file {} {} {}\n",
                f.base.as_deref().unwrap_or("-"),
                f.fin.as_deref().unwrap_or("-"),
                f.path
            ));
        }
        crate::util::write_file(path, s.as_bytes())
    }
}

/// Resume bookkeeping in `<workspace>/.installer/state`.
#[derive(Debug, Default)]
pub struct State {
    pub eboot: Option<String>,
    pub done: Vec<String>,
    pub applied: bool,
}

impl State {
    pub fn load(path: &Path) -> State {
        let mut st = State::default();
        let Ok(txt) = std::fs::read_to_string(path) else {
            return st;
        };
        for line in txt.lines() {
            if let Some(r) = line.strip_prefix("eboot ") {
                st.eboot = Some(r.trim().to_string());
            } else if let Some(r) = line.strip_prefix("done ") {
                st.done.push(r.trim().to_string());
            } else if line.trim() == "applied" {
                st.applied = true;
            }
        }
        st
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut s = String::new();
        if let Some(e) = &self.eboot {
            s.push_str(&format!("eboot {e}\n"));
        }
        for d in &self.done {
            s.push_str(&format!("done {d}\n"));
        }
        if self.applied {
            s.push_str("applied\n");
        }
        crate::util::write_file(path, s.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_roundtrip() {
        let dir = std::env::temp_dir().join(format!("lock-test-{}", std::process::id()));
        let p = dir.join("generated.lock");
        let l = Lock {
            eboot: Some("ab".into()),
            files: vec![
                Entry {
                    path: "a/b c.cpp".into(),
                    base: Some("1".into()),
                    fin: Some("2".into()),
                },
                Entry {
                    path: "x".into(),
                    base: None,
                    fin: Some("3".into()),
                },
                Entry {
                    path: "y".into(),
                    base: Some("4".into()),
                    fin: None,
                },
            ],
        };
        l.save(&p).unwrap();
        let r = Lock::load(&p).unwrap().unwrap();
        assert_eq!(r.eboot, l.eboot);
        assert_eq!(r.files, l.files);
        let _ = std::fs::remove_dir_all(dir);
    }
}
