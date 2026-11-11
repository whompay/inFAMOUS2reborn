//! Line diffs between the pipeline's fresh output and the port's hand-maintained files, written
//! and applied as unified diffs.
//!
//! The files are large (the lifted PPU translation units are ~40 MB and ~1M lines each) and the
//! edits are scattered, so the diff is patience-style: lines that occur exactly once on both sides
//! anchor the alignment (longest increasing subsequence), the gaps between anchors are diffed
//! recursively, and only small leftover gaps fall back to a bounded Myers search. Lines keep their
//! own terminators, so CRLF and LF files round-trip byte for byte.
//!
//! When the pipeline's output and the maintained file use different line endings throughout
//! (Python's text mode writes CRLF on Windows and LF elsewhere, and some maintained files were
//! last written on the other OS), the patch starts with `# eol lf` or `# eol crlf`: the base is
//! converted to that ending first and the hunks are diffed/applied against the converted text.
//! Without this, one changed byte per line would turn into a patch that carries the whole file.
//!
//! Applying is strict: every context and removed line must match exactly at the recorded line
//! number. The installer only applies a patch after checking that the file it starts from has the
//! hash the maintainer's pipeline produced, so a mismatch means a broken patch, not drift.

use std::collections::HashMap;

pub fn split_lines(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, &b) in data.iter().enumerate() {
        if b == b'\n' {
            out.push(&data[start..=i]);
            start = i + 1;
        }
    }
    if start < data.len() {
        out.push(&data[start..]);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tag {
    Equal,
    Delete,
    Insert,
}

/// The edit script: one entry per line of output (Equal/Delete consume an `a` line,
/// Equal/Insert a `b` line).
pub fn diff_lines(a: &[&[u8]], b: &[&[u8]]) -> Vec<Tag> {
    let mut out = Vec::with_capacity(a.len().max(b.len()));
    diff_range(a, b, &mut out);
    out
}

fn push_n(out: &mut Vec<Tag>, t: Tag, n: usize) {
    out.resize(out.len() + n, t);
}

fn diff_range(a: &[&[u8]], b: &[&[u8]], out: &mut Vec<Tag>) {
    // Common prefix / suffix.
    let pre = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    let (a2, b2) = (&a[pre..], &b[pre..]);
    let suf = a2
        .iter()
        .rev()
        .zip(b2.iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let (a3, b3) = (&a2[..a2.len() - suf], &b2[..b2.len() - suf]);
    push_n(out, Tag::Equal, pre);
    if a3.is_empty() || b3.is_empty() {
        push_n(out, Tag::Delete, a3.len());
        push_n(out, Tag::Insert, b3.len());
    } else {
        let anchors = unique_anchors(a3, b3);
        if anchors.is_empty() {
            myers_or_replace(a3, b3, out);
        } else {
            let (mut ai, mut bi) = (0, 0);
            for (x, y) in anchors {
                diff_range(&a3[ai..x], &b3[bi..y], out);
                out.push(Tag::Equal);
                ai = x + 1;
                bi = y + 1;
            }
            diff_range(&a3[ai..], &b3[bi..], out);
        }
    }
    push_n(out, Tag::Equal, suf);
}

/// Pairs (i, j) of lines unique in both slices, reduced to the longest chain increasing in both.
fn unique_anchors(a: &[&[u8]], b: &[&[u8]]) -> Vec<(usize, usize)> {
    // count_a, idx_a, count_b, idx_b
    let mut m: HashMap<&[u8], (u32, usize, u32, usize)> = HashMap::with_capacity(a.len() + b.len());
    for (i, l) in a.iter().enumerate() {
        let e = m.entry(l).or_insert((0, 0, 0, 0));
        e.0 += 1;
        e.1 = i;
    }
    for (j, l) in b.iter().enumerate() {
        if let Some(e) = m.get_mut(l) {
            e.2 += 1;
            e.3 = j;
        }
    }
    let mut pairs: Vec<(usize, usize)> = m
        .values()
        .filter(|e| e.0 == 1 && e.2 == 1)
        .map(|e| (e.1, e.3))
        .collect();
    pairs.sort_unstable();
    // LIS over j (patience sorting with back-pointers).
    let mut tails: Vec<usize> = Vec::new(); // index into pairs
    let mut prev: Vec<usize> = vec![usize::MAX; pairs.len()];
    for (k, &(_, j)) in pairs.iter().enumerate() {
        let pos = tails.partition_point(|&t| pairs[t].1 < j);
        if pos > 0 {
            prev[k] = tails[pos - 1];
        }
        if pos == tails.len() {
            tails.push(k);
        } else {
            tails[pos] = k;
        }
    }
    let mut chain = Vec::with_capacity(tails.len());
    let mut k = *tails.last().unwrap_or(&usize::MAX);
    while k != usize::MAX {
        chain.push(pairs[k]);
        k = prev[k];
    }
    chain.reverse();
    chain
}

/// Myers' O(ND) diff for gaps without unique lines; gives up (whole-gap replace) past a budget.
fn myers_or_replace(a: &[&[u8]], b: &[&[u8]], out: &mut Vec<Tag>) {
    let (n, m) = (a.len() as isize, b.len() as isize);
    let max_d = ((n + m) as usize).min(2_000) as isize;
    let off = max_d + 1;
    let mut v = vec![0isize; (2 * off + 1) as usize];
    let mut trace: Vec<Vec<isize>> = Vec::new();
    let mut found = false;
    'outer: for d in 0..=max_d {
        trace.push(v.clone());
        let mut k = -d;
        while k <= d {
            let mut x =
                if k == -d || (k != d && v[(k - 1 + off) as usize] < v[(k + 1 + off) as usize]) {
                    v[(k + 1 + off) as usize]
                } else {
                    v[(k - 1 + off) as usize] + 1
                };
            let mut y = x - k;
            while x < n && y < m && a[x as usize] == b[y as usize] {
                x += 1;
                y += 1;
            }
            v[(k + off) as usize] = x;
            if x >= n && y >= m {
                found = true;
                break 'outer;
            }
            k += 2;
        }
        if (d as usize) * (n + m) as usize > 50_000_000 {
            break; // budget: keep pathological gaps cheap
        }
    }
    if !found {
        push_n(out, Tag::Delete, a.len());
        push_n(out, Tag::Insert, b.len());
        return;
    }
    // Backtrack: trace[d] is the frontier before step d.
    let mut rev: Vec<Tag> = Vec::new();
    let (mut x, mut y) = (n, m);
    for d in (1..trace.len() as isize).rev() {
        let vd = &trace[d as usize];
        let k = x - y;
        let prev_k =
            if k == -d || (k != d && vd[(k - 1 + off) as usize] < vd[(k + 1 + off) as usize]) {
                k + 1
            } else {
                k - 1
            };
        let prev_x = vd[(prev_k + off) as usize];
        let prev_y = prev_x - prev_k;
        // The snake after the edit, then the edit itself.
        let (sx, sy) = if prev_k == k + 1 {
            (prev_x, prev_y + 1)
        } else {
            (prev_x + 1, prev_y)
        };
        while x > sx && y > sy {
            rev.push(Tag::Equal);
            x -= 1;
            y -= 1;
        }
        if prev_k == k + 1 {
            rev.push(Tag::Insert);
        } else {
            rev.push(Tag::Delete);
        }
        x = prev_x;
        y = prev_y;
    }
    while x > 0 && y > 0 {
        rev.push(Tag::Equal);
        x -= 1;
        y -= 1;
    }
    rev.reverse();
    out.extend(rev);
}

// ---------------------------------------------------------------------------------------------
// Unified diff writer / reader
// ---------------------------------------------------------------------------------------------

const NO_NL: &[u8] = b"\\ No newline at end of file\n";

fn push_line(out: &mut Vec<u8>, prefix: u8, line: &[u8]) {
    out.push(prefix);
    out.extend_from_slice(line);
    if !line.ends_with(b"\n") {
        out.push(b'\n');
        out.extend_from_slice(NO_NL);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Eol {
    Lf,
    Crlf,
}

/// The line ending used by most lines (None if the data has no line breaks).
pub fn dominant_eol(data: &[u8]) -> Option<Eol> {
    let lf = data.iter().filter(|&&b| b == b'\n').count();
    if lf == 0 {
        return None;
    }
    let crlf = data.windows(2).filter(|w| w == b"\r\n").count();
    Some(if crlf * 2 > lf { Eol::Crlf } else { Eol::Lf })
}

/// Rewrite every line ending to `eol` (lone `\r` is left alone).
pub fn convert_eol(data: &[u8], eol: Eol) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 32);
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        if b == b'\r' && data.get(i + 1) == Some(&b'\n') {
            i += 1;
            continue; // the \n below writes the ending
        }
        if b == b'\n' {
            if eol == Eol::Crlf {
                out.push(b'\r');
            }
            out.push(b'\n');
        } else {
            out.push(b);
        }
        i += 1;
    }
    out
}

const EOL_LF: &[u8] = b"# eol lf\n";
const EOL_CRLF: &[u8] = b"# eol crlf\n";

pub struct PatchStats {
    pub hunks: usize,
    pub added_lines: usize,
    pub removed_lines: usize,
    pub added_bytes: usize,
}

/// Unified diff from `old` to `new` with `ctx` lines of context. Empty when identical.
/// If the two files use different line endings throughout, the patch converts `old` first.
pub fn make_patch(old: &[u8], new: &[u8], name: &str, ctx: usize) -> (Vec<u8>, PatchStats) {
    if old != new {
        if let (Some(eo), Some(en)) = (dominant_eol(old), dominant_eol(new)) {
            if eo != en {
                let conv = convert_eol(old, en);
                let (body, st) = make_patch_raw(&conv, new, name, ctx);
                let mut out = if en == Eol::Lf {
                    EOL_LF.to_vec()
                } else {
                    EOL_CRLF.to_vec()
                };
                if body.is_empty() {
                    // Only the line endings differ: a header with no hunks.
                    out.extend_from_slice(format!("--- a/{name}\n+++ b/{name}\n").as_bytes());
                }
                out.extend_from_slice(&body);
                return (out, st);
            }
        }
    }
    make_patch_raw(old, new, name, ctx)
}

fn make_patch_raw(old: &[u8], new: &[u8], name: &str, ctx: usize) -> (Vec<u8>, PatchStats) {
    let a = split_lines(old);
    let b = split_lines(new);
    let tags = diff_lines(&a, &b);
    let mut stats = PatchStats {
        hunks: 0,
        added_lines: 0,
        removed_lines: 0,
        added_bytes: 0,
    };
    // Position of each tag in a and b.
    let mut pos = Vec::with_capacity(tags.len());
    let (mut i, mut j) = (0usize, 0usize);
    for t in &tags {
        pos.push((i, j));
        match t {
            Tag::Equal => {
                i += 1;
                j += 1
            }
            Tag::Delete => i += 1,
            Tag::Insert => j += 1,
        }
    }
    let changes: Vec<usize> = (0..tags.len()).filter(|&k| tags[k] != Tag::Equal).collect();
    let mut out = Vec::new();
    if changes.is_empty() {
        return (out, stats);
    }
    out.extend_from_slice(format!("--- a/{name}\n+++ b/{name}\n").as_bytes());
    let mut c = 0;
    while c < changes.len() {
        // Extend the hunk while the next change is within 2*ctx equal lines.
        let start = changes[c].saturating_sub(ctx);
        let mut last = changes[c];
        while c + 1 < changes.len() && changes[c + 1] - last <= 2 * ctx + 1 {
            c += 1;
            last = changes[c];
        }
        c += 1;
        let end = (last + ctx + 1).min(tags.len());
        let (a0, b0) = pos[start];
        let (mut na, mut nb) = (0, 0);
        for t in &tags[start..end] {
            match t {
                Tag::Equal => {
                    na += 1;
                    nb += 1
                }
                Tag::Delete => na += 1,
                Tag::Insert => nb += 1,
            }
        }
        let sa = if na == 0 { a0 } else { a0 + 1 };
        let sb = if nb == 0 { b0 } else { b0 + 1 };
        out.extend_from_slice(format!("@@ -{sa},{na} +{sb},{nb} @@\n").as_bytes());
        stats.hunks += 1;
        for k in start..end {
            let (pa, pb) = pos[k];
            match tags[k] {
                Tag::Equal => push_line(&mut out, b' ', a[pa]),
                Tag::Delete => {
                    stats.removed_lines += 1;
                    push_line(&mut out, b'-', a[pa])
                }
                Tag::Insert => {
                    stats.added_lines += 1;
                    stats.added_bytes += b[pb].len();
                    push_line(&mut out, b'+', b[pb])
                }
            }
        }
    }
    (out, stats)
}

struct Hunk<'a> {
    old_start: usize,
    old_count: usize,
    lines: Vec<(u8, &'a [u8])>,
}

fn parse_range(s: &str) -> Option<(usize, usize)> {
    let (a, b) = match s.split_once(',') {
        Some((a, b)) => (a, b),
        None => (s, "1"),
    };
    Some((a.parse().ok()?, b.parse().ok()?))
}

fn parse_patch(patch: &[u8]) -> Result<Vec<Hunk<'_>>, String> {
    let raw = split_lines(patch);
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut k = 0;
    while k < raw.len() {
        let l = raw[k];
        k += 1;
        if l.starts_with(b"--- ") || l.starts_with(b"+++ ") || l.starts_with(b"# ") {
            continue;
        }
        if l.starts_with(b"@@ ") {
            let s = String::from_utf8_lossy(l);
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() < 4 || !parts[1].starts_with('-') || !parts[2].starts_with('+') {
                return Err(format!("bad hunk header: {}", s.trim()));
            }
            let (os, oc) = parse_range(&parts[1][1..])
                .ok_or_else(|| format!("bad hunk header: {}", s.trim()))?;
            let (_ns, nc) = parse_range(&parts[2][1..])
                .ok_or_else(|| format!("bad hunk header: {}", s.trim()))?;
            let mut h = Hunk {
                old_start: os,
                old_count: oc,
                lines: Vec::new(),
            };
            let (mut seen_old, mut seen_new) = (0, 0);
            while (seen_old < oc || seen_new < nc) && k < raw.len() {
                let l = raw[k];
                k += 1;
                let (tag, body) = (l[0], &l[1..]);
                let body = if k < raw.len() && raw[k] == NO_NL {
                    k += 1;
                    &body[..body.len() - 1]
                } else {
                    body
                };
                match tag {
                    b' ' => {
                        seen_old += 1;
                        seen_new += 1
                    }
                    b'-' => seen_old += 1,
                    b'+' => seen_new += 1,
                    _ => return Err(format!("bad line in hunk at -{os}")),
                }
                h.lines.push((tag, body));
            }
            if seen_old != oc || seen_new != nc {
                return Err(format!("truncated hunk at -{os}"));
            }
            hunks.push(h);
            continue;
        }
        if l.trim_ascii().is_empty() {
            continue;
        }
        return Err(format!(
            "unexpected line in patch: {}",
            String::from_utf8_lossy(l).trim_end()
        ));
    }
    Ok(hunks)
}

pub fn apply_patch(old: &[u8], patch: &[u8]) -> Result<Vec<u8>, String> {
    let converted;
    let old = if patch.starts_with(EOL_LF) {
        converted = convert_eol(old, Eol::Lf);
        &converted[..]
    } else if patch.starts_with(EOL_CRLF) {
        converted = convert_eol(old, Eol::Crlf);
        &converted[..]
    } else {
        old
    };
    let src = split_lines(old);
    let hunks = parse_patch(patch)?;
    let mut out = Vec::with_capacity(old.len() + patch.len());
    let mut cur = 0usize; // next source line to copy
    for h in &hunks {
        let at = if h.old_count == 0 {
            h.old_start
        } else {
            h.old_start - 1
        };
        if at < cur || at > src.len() {
            return Err(format!(
                "hunk at line {} is out of order or past the end",
                h.old_start
            ));
        }
        for l in &src[cur..at] {
            out.extend_from_slice(l);
        }
        cur = at;
        for &(tag, body) in &h.lines {
            match tag {
                b' ' | b'-' => {
                    if cur >= src.len() || src[cur] != body {
                        return Err(format!(
                            "context mismatch at line {} (hunk at {})",
                            cur + 1,
                            h.old_start
                        ));
                    }
                    if tag == b' ' {
                        out.extend_from_slice(body);
                    }
                    cur += 1;
                }
                _ => out.extend_from_slice(body),
            }
        }
    }
    for l in &src[cur..] {
        out.extend_from_slice(l);
    }
    Ok(out)
}

pub fn is_binary(data: &[u8]) -> bool {
    data.contains(&0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(a: &[u8], b: &[u8]) {
        let (p, _) = make_patch(a, b, "f", 3);
        let got =
            apply_patch(a, &p).unwrap_or_else(|e| panic!("{e}\n{}", String::from_utf8_lossy(&p)));
        assert_eq!(got, b, "patch:\n{}", String::from_utf8_lossy(&p));
    }

    #[test]
    fn small_cases() {
        roundtrip(b"", b"");
        roundtrip(b"", b"a\nb\n");
        roundtrip(b"a\nb\n", b"");
        roundtrip(b"a\nb\nc\n", b"a\nB\nc\n");
        roundtrip(b"a\nb\nc", b"a\nb\nc\n"); // newline added at EOF
        roundtrip(b"a\nb\nc\n", b"a\nb\nc"); // newline removed at EOF
        roundtrip(b"x\r\ny\r\nz\r\n", b"x\r\nY\r\nz\r\nw\r\n"); // CRLF kept
        roundtrip(b"a\na\na\nb\na\na\n", b"a\nb\na\nb\na\n"); // no unique lines
        assert!(make_patch(b"same\n", b"same\n", "f", 3).0.is_empty());
    }

    #[test]
    fn many_scattered_edits() {
        // Pseudo-random file with repeated lines (like generated code) and scattered edits.
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut a = Vec::new();
        for i in 0..20_000 {
            if rnd() % 5 == 0 {
                a.extend_from_slice(b"    ctx->gpr[3] = 0;\n");
            } else {
                a.extend_from_slice(format!("line {i} {}\n", rnd() % 1000).as_bytes());
            }
        }
        let lines = split_lines(&a);
        let mut b = Vec::new();
        for (i, l) in lines.iter().enumerate() {
            match rnd() % 97 {
                0 => {}                                                         // delete
                1 => b.extend_from_slice(format!("inserted {i}\n").as_bytes()), // replace
                2 => {
                    b.extend_from_slice(l);
                    b.extend_from_slice(b"    ctx->gpr[3] = 0;\n"); // insert a repeated line
                }
                _ => b.extend_from_slice(l),
            }
        }
        roundtrip(&a, &b);
        let (p, st) = make_patch(&a, &b, "f", 3);
        assert!(st.hunks > 100);
        assert!(
            p.len() < a.len() / 2,
            "patch should be much smaller than the file"
        );
    }

    #[test]
    fn eol_conversion() {
        // CRLF pipeline output, LF maintained file with one edit: tiny patch, exact result.
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..5000 {
            a.extend_from_slice(format!("line {i}\r\n").as_bytes());
            b.extend_from_slice(
                if i == 2500 {
                    b"edited\n".to_vec()
                } else {
                    format!("line {i}\n").into_bytes()
                }
                .as_slice(),
            );
        }
        let (p, st) = make_patch(&a, &b, "f", 3);
        assert!(p.starts_with(EOL_LF));
        assert_eq!(st.added_lines, 1);
        assert!(p.len() < 200, "{}", String::from_utf8_lossy(&p));
        assert_eq!(apply_patch(&a, &p).unwrap(), b);
        // ... and the other way round.
        let (p, _) = make_patch(&b, &a, "f", 3);
        assert!(p.starts_with(EOL_CRLF));
        assert_eq!(apply_patch(&b, &p).unwrap(), a);
        // Only the endings differ.
        let (p, st) = make_patch(b"x\r\ny\r\n", b"x\ny\n", "f", 3);
        assert_eq!(st.hunks, 0);
        assert_eq!(apply_patch(b"x\r\ny\r\n", &p).unwrap(), b"x\ny\n");
        // Mixed endings in the maintained file survive exactly.
        roundtrip(b"a\r\nb\r\nc\r\nd\r\n", b"a\nb\r\nc\nd\n");
        roundtrip(b"a\nb\nc", b"a\r\nb\r\nc"); // no final newline
    }

    #[test]
    fn strict_apply_rejects_wrong_base() {
        let (p, _) = make_patch(b"a\nb\nc\n", b"a\nX\nc\n", "f", 1);
        assert!(apply_patch(b"a\nq\nc\n", &p).is_err());
    }
}
