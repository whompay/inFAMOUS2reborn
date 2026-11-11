//! A small TOML reader: enough of the format for the installer manifest.
//!
//! Supported: comments, `[table]` / `[a.b]` headers, `[[array.of.tables]]`, `key = value` with bare or
//! quoted keys, basic and literal strings (single line), integers, booleans, arrays (may span lines,
//! trailing comma allowed) and inline tables. Not supported (and rejected with a line number):
//! multi-line strings, floats, dates, dotted keys on the left of `=`.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Int(i64),
    Bool(bool),
    Arr(Vec<Value>),
    Table(Table),
}

/// Insertion-ordered table (the manifest's env blocks keep the order they were written in).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Table(pub Vec<(String, Value)>);

impl Table {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.0.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }
}

#[derive(Debug)]
pub struct Error {
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.msg)
    }
}
impl std::error::Error for Error {}

struct P<'a> {
    s: &'a [u8],
    i: usize,
    line: usize,
}

impl<'a> P<'a> {
    fn err<T>(&self, msg: impl Into<String>) -> Result<T, Error> {
        Err(Error {
            line: self.line,
            msg: msg.into(),
        })
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.i += 1;
        if c == b'\n' {
            self.line += 1;
        }
        Some(c)
    }
    /// Spaces and tabs only.
    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r')) {
            self.bump();
        }
    }
    /// Whitespace, newlines and comments (inside arrays).
    fn ws_nl(&mut self) {
        loop {
            match self.peek() {
                Some(b' ' | b'\t' | b'\r' | b'\n') => {
                    self.bump();
                }
                Some(b'#') => self.comment(),
                _ => break,
            }
        }
    }
    fn comment(&mut self) {
        while let Some(c) = self.peek() {
            if c == b'\n' {
                break;
            }
            self.bump();
        }
    }
    /// After a value or header: optional comment, then newline or EOF.
    fn eol(&mut self) -> Result<(), Error> {
        self.ws();
        if self.peek() == Some(b'#') {
            self.comment();
        }
        match self.peek() {
            None => Ok(()),
            Some(b'\n') => {
                self.bump();
                Ok(())
            }
            Some(c) => self.err(format!("unexpected '{}' after value", c as char)),
        }
    }

    fn key(&mut self) -> Result<String, Error> {
        match self.peek() {
            Some(b'"') => self.basic_string(),
            Some(b'\'') => self.literal_string(),
            _ => {
                let start = self.i;
                while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
                {
                    self.bump();
                }
                if start == self.i {
                    return self.err("expected a key");
                }
                Ok(String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
            }
        }
    }

    fn basic_string(&mut self) -> Result<String, Error> {
        self.bump(); // opening quote
        let mut out = Vec::new();
        loop {
            if matches!(self.peek(), None | Some(b'\n')) {
                return self.err("unterminated string");
            }
            match self.bump() {
                None | Some(b'\n') => unreachable!(),
                Some(b'"') => break,
                Some(b'\\') => match self.bump() {
                    Some(b'"') => out.push(b'"'),
                    Some(b'\\') => out.push(b'\\'),
                    Some(b'n') => out.push(b'\n'),
                    Some(b't') => out.push(b'\t'),
                    Some(b'r') => out.push(b'\r'),
                    Some(b'u') => {
                        let h = self.s.get(self.i..self.i + 4).ok_or(Error {
                            line: self.line,
                            msg: "bad \\u".into(),
                        })?;
                        let cp = u32::from_str_radix(std::str::from_utf8(h).unwrap_or("x"), 16)
                            .map_err(|_| Error {
                                line: self.line,
                                msg: "bad \\u escape".into(),
                            })?;
                        self.i += 4;
                        let ch = char::from_u32(cp).ok_or(Error {
                            line: self.line,
                            msg: "bad code point".into(),
                        })?;
                        let mut b = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                    }
                    _ => return self.err("unknown escape in string"),
                },
                Some(c) => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| Error {
            line: self.line,
            msg: "string is not UTF-8".into(),
        })
    }

    fn literal_string(&mut self) -> Result<String, Error> {
        self.bump();
        let start = self.i;
        loop {
            if matches!(self.peek(), None | Some(b'\n')) {
                return self.err("unterminated string");
            }
            match self.bump() {
                None | Some(b'\n') => unreachable!(),
                Some(b'\'') => break,
                _ => {}
            }
        }
        Ok(String::from_utf8_lossy(&self.s[start..self.i - 1]).into_owned())
    }

    fn value(&mut self) -> Result<Value, Error> {
        match self.peek() {
            Some(b'"') => {
                if self.s[self.i..].starts_with(b"\"\"\"") {
                    return self.err("multi-line strings are not supported");
                }
                Ok(Value::Str(self.basic_string()?))
            }
            Some(b'\'') => Ok(Value::Str(self.literal_string()?)),
            Some(b'[') => {
                self.bump();
                let mut items = Vec::new();
                loop {
                    self.ws_nl();
                    if self.peek() == Some(b']') {
                        self.bump();
                        break;
                    }
                    items.push(self.value()?);
                    self.ws_nl();
                    match self.bump() {
                        Some(b',') => continue,
                        Some(b']') => break,
                        _ => return self.err("expected ',' or ']' in array"),
                    }
                }
                Ok(Value::Arr(items))
            }
            Some(b'{') => {
                self.bump();
                let mut t = Table::default();
                self.ws();
                if self.peek() == Some(b'}') {
                    self.bump();
                    return Ok(Value::Table(t));
                }
                loop {
                    self.ws();
                    let k = self.key()?;
                    self.ws();
                    if self.bump() != Some(b'=') {
                        return self.err("expected '=' in inline table");
                    }
                    self.ws();
                    let v = self.value()?;
                    if t.get(&k).is_some() {
                        return self.err(format!("duplicate key '{k}'"));
                    }
                    t.0.push((k, v));
                    self.ws();
                    match self.bump() {
                        Some(b',') => continue,
                        Some(b'}') => break,
                        _ => return self.err("expected ',' or '}' in inline table"),
                    }
                }
                Ok(Value::Table(t))
            }
            Some(b't') if self.s[self.i..].starts_with(b"true") => {
                self.i += 4;
                Ok(Value::Bool(true))
            }
            Some(b'f') if self.s[self.i..].starts_with(b"false") => {
                self.i += 5;
                Ok(Value::Bool(false))
            }
            Some(c) if c.is_ascii_digit() || c == b'-' || c == b'+' => {
                let start = self.i;
                self.bump();
                while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == b'_') {
                    self.bump();
                }
                let txt: String = String::from_utf8_lossy(&self.s[start..self.i]).replace('_', "");
                let (neg, body) = match txt.strip_prefix('-') {
                    Some(b) => (true, b.to_string()),
                    None => (false, txt.trim_start_matches('+').to_string()),
                };
                let v = if let Some(h) = body.strip_prefix("0x") {
                    i64::from_str_radix(h, 16)
                } else {
                    body.parse::<i64>()
                };
                match v {
                    Ok(v) => Ok(Value::Int(if neg { -v } else { v })),
                    Err(_) => self.err(format!("unsupported number '{txt}'")),
                }
            }
            _ => self.err("expected a value"),
        }
    }
}

/// Walk/create the table at `path`. For an array of tables the LAST element is used.
fn descend<'t>(root: &'t mut Table, path: &[String], line: usize) -> Result<&'t mut Table, Error> {
    let mut cur = root;
    for seg in path {
        if cur.get(seg).is_none() {
            cur.0.push((seg.clone(), Value::Table(Table::default())));
        }
        cur = match cur.get_mut(seg).unwrap() {
            Value::Table(t) => t,
            Value::Arr(a) => match a.last_mut() {
                Some(Value::Table(t)) => t,
                _ => {
                    return Err(Error {
                        line,
                        msg: format!("'{seg}' is not a table"),
                    })
                }
            },
            _ => {
                return Err(Error {
                    line,
                    msg: format!("'{seg}' is not a table"),
                })
            }
        };
    }
    Ok(cur)
}

pub fn parse(src: &str) -> Result<Table, Error> {
    let mut p = P {
        s: src.as_bytes(),
        i: 0,
        line: 1,
    };
    let mut root = Table::default();
    let mut current: Vec<String> = Vec::new();
    loop {
        p.ws_nl();
        let Some(c) = p.peek() else { break };
        if c == b'[' {
            p.bump();
            let is_array = p.peek() == Some(b'[');
            if is_array {
                p.bump();
            }
            let mut path = Vec::new();
            loop {
                p.ws();
                path.push(p.key()?);
                p.ws();
                match p.peek() {
                    Some(b'.') => {
                        p.bump();
                    }
                    Some(b']') => break,
                    _ => return p.err("bad table header"),
                }
            }
            p.bump();
            if is_array && p.bump() != Some(b']') {
                return p.err("expected ']]'");
            }
            p.eol()?;
            let line = p.line;
            if is_array {
                let (last, parent) = path.split_last().unwrap();
                let t = descend(&mut root, parent, line)?;
                match t.get_mut(last) {
                    None => t.0.push((
                        last.clone(),
                        Value::Arr(vec![Value::Table(Table::default())]),
                    )),
                    Some(Value::Arr(a)) => a.push(Value::Table(Table::default())),
                    Some(_) => return p.err(format!("'{last}' is not an array of tables")),
                }
            } else {
                descend(&mut root, &path, line)?;
            }
            current = path;
            continue;
        }
        let key = p.key()?;
        p.ws();
        if p.peek() == Some(b'.') {
            return p.err("dotted keys are not supported; use a [table] header");
        }
        if p.bump() != Some(b'=') {
            return p.err(format!("expected '=' after '{key}'"));
        }
        p.ws();
        let v = p.value()?;
        p.eol()?;
        let line = p.line;
        let t = descend(&mut root, &current, line)?;
        if t.get(&key).is_some() {
            return Err(Error {
                line: line - 1,
                msg: format!("duplicate key '{key}'"),
            });
        }
        t.0.push((key, v));
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_manifest_shapes() {
        let src = r#"
# comment
title = "inFamous"   # trailing
[game]
ids = ["BCES00220", 'x\y',
       ]   # array over lines
n = 0x10
on = true
[[step]]
id = "a"
env = { A = "1", "B C" = "two" }
[[step]]
id = "b"
run = [
  "{python}",  # comment inside
  "x",
]
"#;
        let t = parse(src).unwrap();
        assert_eq!(t.get("title"), Some(&Value::Str("inFamous".into())));
        let Value::Table(g) = t.get("game").unwrap() else {
            panic!()
        };
        assert_eq!(g.get("n"), Some(&Value::Int(16)));
        assert_eq!(
            g.get("ids"),
            Some(&Value::Arr(vec![
                Value::Str("BCES00220".into()),
                Value::Str("x\\y".into())
            ]))
        );
        let Value::Arr(steps) = t.get("step").unwrap() else {
            panic!()
        };
        assert_eq!(steps.len(), 2);
        let Value::Table(s0) = &steps[0] else {
            panic!()
        };
        let Value::Table(env) = s0.get("env").unwrap() else {
            panic!()
        };
        assert_eq!(env.get("B C"), Some(&Value::Str("two".into())));
    }

    #[test]
    fn reports_line_numbers() {
        let e = parse("a = 1\nb = \"open\n").unwrap_err();
        assert_eq!(e.line, 2);
        assert!(parse("a = 1\na = 2\n").is_err());
    }
}
