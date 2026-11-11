//! `installer/infamous2.toml`: everything game- and port-specific lives there, so the steps can be
//! changed without rebuilding the installer.

use crate::toml::{self, Table, Value};
use crate::util::{Ctx, Error, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Step {
    pub id: String,
    pub title: String,
    /// Program + arguments (placeholders expanded). Mutually exclusive with `builtin`.
    pub run: Vec<String>,
    /// Built-in action implemented by the installer (see pipeline.rs).
    pub builtin: Option<String>,
    pub args: Vec<String>,
    /// Run once per file matching this glob (relative to the workspace); `{file}` is the match.
    pub for_each: Option<String>,
    pub cwd: Option<String>,
    pub env: Vec<(String, String)>,
    pub mkdirs: Vec<String>,
    /// A failing optional step is reported and skipped; anything that depends on its output
    /// will then show up as a hash mismatch (or a missing asset at run time).
    pub optional: bool,
    /// Files this step writes (globs, workspace-relative). If an optional step fails, the
    /// installer tolerates these being missing instead of failing the whole install.
    pub outputs: Vec<String>,
    pub note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Manifest {
    pub game_title: String,
    pub title_ids: Vec<String>,
    pub eboot_sha256: Vec<String>,
    pub link_name: Option<String>,

    pub port: String,
    pub toolkit: String,

    pub py_min: (u32, u32),
    pub py_requirements: Vec<String>,
    pub py_packages: Vec<String>,

    pub gen_include: Vec<String>,
    pub gen_exclude: Vec<String>,
    pub full_patch_ok: Vec<String>,
    /// Generated files whose bytes legitimately vary between machines (drawn with the system's
    /// fonts / image library). Never published, never patched, never hash-checked: the player
    /// keeps whatever their pipeline produced.
    pub gen_loose: Vec<String>,

    pub steps: Vec<Step>,

    pub build_configure: Vec<String>,
    pub build_compile: Vec<String>,
    pub build_configure_unix: Vec<String>,
    pub build_compile_unix: Vec<String>,
    pub build_exe: String,
    pub build_exe_unix: String,
    pub build_log: String,

    pub launch_name: String,
    pub launch_cwd: String,
    pub launch_env: Vec<(String, String)>,
    pub launch_stdout: Option<String>,
    pub launch_stderr: Option<String>,

    pub export_exclude: Vec<String>,
    pub export_keep: Vec<String>,
    /// `[export] forbid`: files that must never be published or packed into a setup (private
    /// mods, notes). `export` / `package` refuse to run while any of them would be included.
    pub export_forbid: Vec<String>,

    /// `[update]`: where `infamous2-installer update` fetches new versions (a GitHub repository
    /// holding the `export` tree). `None` = updating is not configured.
    pub update_repo: Option<String>,
    pub update_branch: String,
    /// Folder inside the repository that holds the tree ("" = the repository root).
    pub update_path: String,
    /// `[update] obsolete`: files an older setup or update left behind that the repository no
    /// longer has; `update` deletes them wherever they still exist.
    pub update_obsolete: Vec<String>,
}

fn table<'a>(t: &'a Table, key: &str) -> Result<&'a Table> {
    match t.get(key) {
        Some(Value::Table(x)) => Ok(x),
        Some(_) => Err(Error(format!("[{key}] must be a table"))),
        None => Err(Error(format!("missing [{key}] section"))),
    }
}

fn opt_table<'a>(t: &'a Table, key: &str) -> Result<Option<&'a Table>> {
    match t.get(key) {
        None => Ok(None),
        Some(Value::Table(x)) => Ok(Some(x)),
        Some(_) => Err(Error(format!("[{key}] must be a table"))),
    }
}

fn s(t: &Table, sect: &str, key: &str) -> Result<String> {
    match t.get(key) {
        Some(Value::Str(v)) => Ok(v.clone()),
        Some(_) => Err(Error(format!("{sect}.{key} must be a string"))),
        None => Err(Error(format!("missing {sect}.{key}"))),
    }
}

fn opt_s(t: &Table, sect: &str, key: &str) -> Result<Option<String>> {
    match t.get(key) {
        None => Ok(None),
        Some(Value::Str(v)) => Ok(Some(v.clone())),
        Some(_) => Err(Error(format!("{sect}.{key} must be a string"))),
    }
}

fn list(t: &Table, sect: &str, key: &str) -> Result<Vec<String>> {
    match t.get(key) {
        None => Ok(Vec::new()),
        Some(Value::Arr(a)) => a
            .iter()
            .map(|v| match v {
                Value::Str(x) => Ok(x.clone()),
                Value::Int(i) => Ok(i.to_string()),
                _ => Err(Error(format!("{sect}.{key} must be a list of strings"))),
            })
            .collect(),
        Some(_) => Err(Error(format!("{sect}.{key} must be a list"))),
    }
}

fn env(t: &Table, sect: &str) -> Result<Vec<(String, String)>> {
    match t.get("env") {
        None => Ok(Vec::new()),
        Some(Value::Table(e)) => {
            e.0.iter()
                .map(|(k, v)| match v {
                    Value::Str(x) => Ok((k.clone(), x.clone())),
                    Value::Int(i) => Ok((k.clone(), i.to_string())),
                    Value::Bool(b) => Ok((k.clone(), if *b { "1".into() } else { "0".into() })),
                    _ => Err(Error(format!("{sect}.env.{k} must be a string"))),
                })
                .collect()
        }
        Some(_) => Err(Error(format!("{sect}.env must be an inline table"))),
    }
}

fn boolean(t: &Table, sect: &str, key: &str) -> Result<bool> {
    match t.get(key) {
        None => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(Error(format!("{sect}.{key} must be true or false"))),
    }
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Manifest> {
        let src = std::fs::read_to_string(path).ctx(|| format!("reading {}", path.display()))?;
        Self::parse(&src).map_err(|e| Error(format!("{}: {e}", path.display())))
    }

    pub fn parse(src: &str) -> Result<Manifest> {
        let root = toml::parse(src)?;

        let game = table(&root, "game")?;
        let layout = table(&root, "layout")?;
        let python = table(&root, "python")?;
        let generated = table(&root, "generated")?;
        let build = table(&root, "build")?;
        let launch = table(&root, "launch")?;
        let export = opt_table(&root, "export")?;
        let update = opt_table(&root, "update")?;

        let py_min_s = opt_s(python, "python", "min_version")?.unwrap_or_else(|| "3.10".into());
        let py_min = {
            let mut it = py_min_s.split('.').map(|x| x.parse::<u32>().unwrap_or(0));
            (it.next().unwrap_or(3), it.next().unwrap_or(10))
        };

        let mut steps = Vec::new();
        match root.get("step") {
            None => {}
            Some(Value::Arr(a)) => {
                for (i, v) in a.iter().enumerate() {
                    let Value::Table(t) = v else {
                        crate::bail!("[[step]] #{} is not a table", i + 1)
                    };
                    let sect = format!("step #{}", i + 1);
                    let id = s(t, &sect, "id")?;
                    let sect = format!("step '{id}'");
                    let run = list(t, &sect, "run")?;
                    let builtin = opt_s(t, &sect, "builtin")?;
                    if run.is_empty() == builtin.is_none() {
                        crate::bail!("{sect}: give exactly one of `run` or `builtin`");
                    }
                    for k in t.keys() {
                        if ![
                            "id", "title", "run", "builtin", "args", "for_each", "cwd", "env",
                            "mkdirs", "optional", "outputs", "note",
                        ]
                        .contains(&k)
                        {
                            crate::bail!("{sect}: unknown key '{k}'");
                        }
                    }
                    if steps.iter().any(|x: &Step| x.id == id) {
                        crate::bail!("duplicate step id '{id}'");
                    }
                    steps.push(Step {
                        title: opt_s(t, &sect, "title")?.unwrap_or_else(|| id.clone()),
                        run,
                        builtin,
                        args: list(t, &sect, "args")?,
                        for_each: opt_s(t, &sect, "for_each")?,
                        cwd: opt_s(t, &sect, "cwd")?,
                        env: env(t, &sect)?,
                        mkdirs: list(t, &sect, "mkdirs")?,
                        optional: boolean(t, &sect, "optional")?,
                        outputs: list(t, &sect, "outputs")?,
                        note: opt_s(t, &sect, "note")?,
                        id,
                    });
                }
            }
            Some(_) => crate::bail!("`step` must be written as [[step]] tables"),
        }

        let empty = Table::default();
        let export = export.unwrap_or(&empty);
        let update = update.unwrap_or(&empty);
        Ok(Manifest {
            game_title: s(game, "game", "title")?,
            title_ids: list(game, "game", "title_ids")?,
            eboot_sha256: list(game, "game", "eboot_sha256")?
                .into_iter()
                .map(|x| x.to_ascii_lowercase())
                .collect(),
            link_name: opt_s(game, "game", "link_name")?,
            port: s(layout, "layout", "port")?,
            toolkit: s(layout, "layout", "toolkit")?,
            py_min,
            py_requirements: list(python, "python", "requirements")?,
            py_packages: list(python, "python", "packages")?,
            gen_include: list(generated, "generated", "include")?,
            gen_exclude: list(generated, "generated", "exclude")?,
            full_patch_ok: list(generated, "generated", "full_patch_ok")?,
            gen_loose: list(generated, "generated", "loose")?,
            steps,
            build_configure: list(build, "build", "configure")?,
            build_compile: list(build, "build", "compile")?,
            build_configure_unix: list(build, "build", "configure_unix")?,
            build_compile_unix: list(build, "build", "compile_unix")?,
            build_exe: s(build, "build", "exe")?,
            build_exe_unix: opt_s(build, "build", "exe_unix")?
                .unwrap_or_else(|| s(build, "build", "exe").unwrap_or_default()),
            build_log: opt_s(build, "build", "log")?.unwrap_or_else(|| "{port}/build.log".into()),
            launch_name: s(launch, "launch", "name")?,
            launch_cwd: opt_s(launch, "launch", "cwd")?.unwrap_or_else(|| "{port}".into()),
            launch_env: env(launch, "launch")?,
            launch_stdout: opt_s(launch, "launch", "stdout")?,
            launch_stderr: opt_s(launch, "launch", "stderr")?,
            export_exclude: list(export, "export", "exclude")?,
            export_keep: list(export, "export", "keep")?,
            export_forbid: list(export, "export", "forbid")?,
            update_repo: opt_s(update, "update", "repo")?
                .map(|r| r.trim().trim_end_matches('/').to_string())
                .filter(|r| !r.is_empty()),
            update_branch: opt_s(update, "update", "branch")?
                .filter(|b| !b.trim().is_empty())
                .unwrap_or_else(|| "main".into()),
            update_path: opt_s(update, "update", "path")?
                .unwrap_or_default()
                .trim_matches('/')
                .to_string(),
            update_obsolete: list(update, "update", "obsolete")?,
        })
    }

    /// Is this (workspace-relative) file produced from the user's game files?
    pub fn is_loose(&self, rel: &str) -> bool {
        crate::util::any_match(&self.gen_loose, rel)
    }

    pub fn is_generated(&self, rel: &str) -> bool {
        crate::util::any_match(&self.gen_include, rel)
            && !crate::util::any_match(&self.gen_exclude, rel)
    }
}
