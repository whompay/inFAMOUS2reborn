//! Small helpers: errors, console output, globs, directory walks, running programs.

use std::fmt;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<E: std::error::Error> From<E> for Error {
    fn from(e: E) -> Self {
        Error(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! bail {
    ($($t:tt)*) => { return Err($crate::util::Error(format!($($t)*))) };
}

/// Attach context to an error: `fs::read(p).ctx(|| format!("reading {}", p.display()))?`
pub trait Ctx<T> {
    fn ctx<S: Into<String>>(self, f: impl FnOnce() -> S) -> Result<T>;
}
impl<T, E: fmt::Display> Ctx<T> for std::result::Result<T, E> {
    fn ctx<S: Into<String>>(self, f: impl FnOnce() -> S) -> Result<T> {
        self.map_err(|e| Error(format!("{}: {}", f().into(), e)))
    }
}
impl<T> Ctx<T> for Option<T> {
    fn ctx<S: Into<String>>(self, f: impl FnOnce() -> S) -> Result<T> {
        self.ok_or_else(|| Error(f().into()))
    }
}

// ---------------------------------------------------------------------------------------------
// Console
// ---------------------------------------------------------------------------------------------

pub fn heading(s: &str) {
    println!();
    println!("==> {s}");
}
pub fn info(s: &str) {
    println!("    {s}");
}
pub fn warn(s: &str) {
    println!("    WARNING: {s}");
}

pub fn prompt(question: &str) -> String {
    print!("{question}");
    let _ = io::stdout().flush();
    let mut line = String::new();
    let _ = io::stdin().lock().read_line(&mut line);
    // Paths dragged into a Windows console arrive wrapped in quotes.
    line.trim().trim_matches('"').to_string()
}

// ---------------------------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------------------------

/// `a/b/c` relative path with forward slashes (the form used in the manifest, lock and patches).
pub fn rel_slash(root: &Path, p: &Path) -> Option<String> {
    let r = p.strip_prefix(root).ok()?;
    let mut parts = Vec::new();
    for c in r.components() {
        match c {
            Component::Normal(s) => parts.push(s.to_string_lossy().into_owned()),
            _ => return None,
        }
    }
    Some(parts.join("/"))
}

/// Join a forward-slash relative path onto a root.
pub fn join_slash(root: &Path, rel: &str) -> PathBuf {
    let mut p = root.to_path_buf();
    for seg in rel.split('/').filter(|s| !s.is_empty()) {
        p.push(seg);
    }
    p
}

/// Absolute path without requiring it to exist; strips Windows' `\\?\` prefix so the result
/// can be handed to cmd.exe, CMake and Python unchanged.
pub fn absolute(p: &Path) -> PathBuf {
    let abs = if let Ok(c) = fs::canonicalize(p) {
        c
    } else if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|d| d.join(p))
            .unwrap_or_else(|_| p.to_path_buf())
    };
    let s = abs.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = s.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        abs
    }
}

pub fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

// ---------------------------------------------------------------------------------------------
// Globs: `**` = any number of path segments, `*` / `?` inside one segment. Patterns are matched
// against forward-slash relative paths. A pattern ending in `/` matches a directory and
// everything below it.
// ---------------------------------------------------------------------------------------------

fn seg_match(p: &[u8], s: &[u8]) -> bool {
    // Iterative wildcard match with backtracking on the last '*'.
    let (mut pi, mut si) = (0, 0);
    let (mut star, mut mark) = (usize::MAX, 0);
    while si < s.len() {
        if pi < p.len() && (p[pi] == b'?' || p[pi].eq_ignore_ascii_case(&s[si])) {
            pi += 1;
            si += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star = pi;
            mark = si;
            pi += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            si = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

fn segs_match(p: &[&str], s: &[&str]) -> bool {
    match p.split_first() {
        None => s.is_empty(),
        Some((&"**", rest)) => (0..=s.len()).any(|k| segs_match(rest, &s[k..])),
        Some((first, rest)) => {
            !s.is_empty()
                && seg_match(first.as_bytes(), s[0].as_bytes())
                && segs_match(rest, &s[1..])
        }
    }
}

/// Case-insensitive (the port is built on Windows, whose file system is).
pub fn glob_match(pattern: &str, path: &str) -> bool {
    let pattern = if let Some(dir) = pattern.strip_suffix('/') {
        format!("{dir}/**")
    } else {
        pattern.to_string()
    };
    let p: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
    let s: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    segs_match(&p, &s)
}

pub fn any_match(patterns: &[String], path: &str) -> bool {
    patterns.iter().any(|p| glob_match(p, path))
}

/// Is `pattern` able to match something at or below directory `dir`? Used to prune walks.
pub fn may_match_below(pattern: &str, dir: &str) -> bool {
    let pattern = if let Some(d) = pattern.strip_suffix('/') {
        format!("{d}/**")
    } else {
        pattern.to_string()
    };
    let p: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
    let d: Vec<&str> = dir.split('/').filter(|s| !s.is_empty()).collect();
    fn rec(p: &[&str], d: &[&str]) -> bool {
        if d.is_empty() {
            return true;
        }
        match p.split_first() {
            None => false,
            Some((&"**", _)) => true,
            Some((first, rest)) => {
                seg_match(first.as_bytes(), d[0].as_bytes()) && rec(rest, &d[1..])
            }
        }
    }
    rec(&p, &d)
}

/// Every regular file under `root` (relative, forward slashes, sorted). Symlinks and
/// junctions are never followed (the game folder is linked into the workspace), and directory
/// names in `skip_dirs` are skipped wherever they appear.
pub fn walk_files(
    root: &Path,
    skip_dirs: &[&str],
    prune: &dyn Fn(&str) -> bool,
) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => bail!("reading directory {}: {e}", dir.display()),
        };
        for ent in rd {
            let ent = ent?;
            let ft = ent.file_type()?;
            let path = ent.path();
            let Some(rel) = rel_slash(root, &path) else {
                continue;
            };
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                let name = ent.file_name().to_string_lossy().into_owned();
                if skip_dirs.iter().any(|s| s.eq_ignore_ascii_case(&name)) || !prune(&rel) {
                    continue;
                }
                stack.push(path);
            } else if ft.is_file() {
                out.push(rel);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Files under `root` that match `include` and not `exclude`.
pub fn select_files(
    root: &Path,
    include: &[String],
    exclude: &[String],
    skip_dirs: &[&str],
) -> Result<Vec<String>> {
    let prune = |dir: &str| include.iter().any(|p| may_match_below(p, dir));
    let files = walk_files(root, skip_dirs, &prune)?;
    Ok(files
        .into_iter()
        .filter(|f| any_match(include, f) && !any_match(exclude, f))
        .collect())
}

pub fn ensure_parent(p: &Path) -> Result<()> {
    if let Some(d) = p.parent() {
        fs::create_dir_all(d).ctx(|| format!("creating {}", d.display()))?;
    }
    Ok(())
}

pub fn copy_file(src: &Path, dst: &Path) -> Result<()> {
    ensure_parent(dst)?;
    fs::copy(src, dst).ctx(|| format!("copying {} -> {}", src.display(), dst.display()))?;
    Ok(())
}

/// Write atomically enough for our purposes: temp file in the same directory, then rename.
pub fn write_file(p: &Path, data: &[u8]) -> Result<()> {
    ensure_parent(p)?;
    let tmp = p.with_extension("installer-tmp");
    fs::write(&tmp, data).ctx(|| format!("writing {}", tmp.display()))?;
    if p.exists() {
        fs::remove_file(p).ctx(|| format!("replacing {}", p.display()))?;
    }
    fs::rename(&tmp, p).ctx(|| format!("renaming {} -> {}", tmp.display(), p.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------------------------

pub fn show_cmd(cmd: &Command) -> String {
    let mut s = cmd.get_program().to_string_lossy().into_owned();
    for a in cmd.get_args() {
        let a = a.to_string_lossy();
        if a.contains(' ') || a.is_empty() {
            s.push_str(&format!(" \"{a}\""));
        } else {
            s.push(' ');
            s.push_str(&a);
        }
    }
    s
}

/// Run with inherited stdio; error on a non-zero exit.
pub fn run(cmd: &mut Command) -> Result<()> {
    let shown = show_cmd(cmd);
    let status = cmd.status().ctx(|| format!("could not start: {shown}"))?;
    if !status.success() {
        bail!(
            "command failed ({}): {shown}",
            status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".into())
        );
    }
    Ok(())
}

/// Run and capture stdout (stderr discarded); None if it can't start or fails.
pub fn capture(cmd: &mut Command) -> Option<String> {
    #[cfg(windows)]
    {
        // No console window flashing up when the setup program (a GUI app) checks tools.
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Installed physical memory, if the OS tells us.
pub fn total_ram_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            len: u32,
            load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page: u64,
            avail_page: u64,
            total_virt: u64,
            avail_virt: u64,
            avail_ext: u64,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GlobalMemoryStatusEx(p: *mut MemoryStatusEx) -> i32;
        }
        let mut m = MemoryStatusEx {
            len: std::mem::size_of::<MemoryStatusEx>() as u32,
            load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page: 0,
            avail_page: 0,
            total_virt: 0,
            avail_virt: 0,
            avail_ext: 0,
        };
        // SAFETY: m is a properly sized MEMORYSTATUSEX with dwLength set.
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 {
            return Some(m.total_phys);
        }
        None
    }
    #[cfg(not(windows))]
    {
        let s = fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = s
            .lines()
            .find(|l| l.starts_with("MemTotal:"))?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()?;
        Some(kb * 1024)
    }
}

/// Free bytes on the volume holding `p` (or its nearest existing parent).
pub fn free_space(p: &Path) -> Option<u64> {
    let mut d = absolute(p);
    while !d.exists() {
        d = d.parent()?.to_path_buf();
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        #[link(name = "kernel32")]
        extern "system" {
            fn GetDiskFreeSpaceExW(
                dir: *const u16,
                avail: *mut u64,
                total: *mut u64,
                free: *mut u64,
            ) -> i32;
        }
        let w: Vec<u16> = d.as_os_str().encode_wide().chain(Some(0)).collect();
        let (mut a, mut t, mut f) = (0u64, 0u64, 0u64);
        // SAFETY: w is NUL-terminated; the out pointers are valid.
        if unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut a, &mut t, &mut f) } != 0 {
            return Some(a);
        }
        None
    }
    #[cfg(not(windows))]
    {
        let out = Command::new("df").arg("-Pk").arg(&d).output().ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        let kb: u64 = s.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()?;
        Some(kb * 1024)
    }
}

/// Find an executable on a PATH-style string (PATHEXT handled on Windows).
pub fn find_in_path(name: &str, path_var: Option<&str>) -> Option<PathBuf> {
    let path_var = match path_var {
        Some(p) => p.to_string(),
        None => std::env::var("PATH").ok()?,
    };
    let sep = if cfg!(windows) { ';' } else { ':' };
    let exts: Vec<String> = if cfg!(windows) && Path::new(name).extension().is_none() {
        vec![".exe".into(), ".cmd".into(), ".bat".into()]
    } else {
        vec![String::new()]
    };
    for dir in path_var.split(sep).filter(|d| !d.is_empty()) {
        for e in &exts {
            let cand = Path::new(dir.trim_matches('"')).join(format!("{name}{e}"));
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

pub fn human_bytes(n: u64) -> String {
    if n >= 1 << 30 {
        format!("{:.1} GB", n as f64 / (1u64 << 30) as f64)
    } else if n >= 1 << 20 {
        format!("{:.1} MB", n as f64 / (1u64 << 20) as f64)
    } else if n >= 1 << 10 {
        format!("{:.1} KB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        assert!(glob_match(
            "infamous-recomp/recompiled/**",
            "infamous-recomp/recompiled/ppu_recomp_000.cpp"
        ));
        assert!(glob_match(
            "infamous-recomp/recompiled/",
            "infamous-recomp/recompiled/a/b.cpp"
        ));
        assert!(glob_match(
            "**/*.pre_v200",
            "infamous-recomp/spu/jobmod_x/spu_recomp_rc.c.pre_v200"
        ));
        assert!(glob_match("**/*.log", "build.log"));
        assert!(glob_match(
            "infamous-recomp/spu/*/spu_recomp_rc.c",
            "infamous-recomp/spu/wwsplug/spu_recomp_rc.c"
        ));
        assert!(!glob_match(
            "infamous-recomp/spu/*/spu_recomp_rc.c",
            "infamous-recomp/spu/spu_recomp_rc.c"
        ));
        assert!(glob_match(
            "BCES00220_guest_memory_*/",
            "BCES00220_guest_memory_2026/x.bin"
        ));
        assert!(glob_match("**/CMakeLists.txt", "ps3recomp/CMakeLists.txt"));
        assert!(!glob_match(
            "infamous-recomp/*.txt",
            "infamous-recomp/tools/x.txt"
        ));
        assert!(glob_match(
            "infamous-recomp/build*/",
            "infamous-recomp/build-fast/x.obj"
        ));
        assert!(glob_match("a/*b*c", "a/xxbyyc"));
        assert!(!glob_match("a/*b*c", "a/xxbyy"));
        assert!(may_match_below("infamous-recomp/spu/**", "infamous-recomp"));
        assert!(!may_match_below("infamous-recomp/spu/**", "ps3recomp"));
        assert!(may_match_below("**/*.log", "anything/deep"));
        assert!(may_match_below(
            "infamous-recomp/spu/",
            "infamous-recomp/spu/jobmods"
        ));
    }
}
