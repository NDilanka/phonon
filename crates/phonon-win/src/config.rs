//! Windows controls for the pinned runtimes, not arbitrary server args.
//!
//! PHONON_WIN_ASR_THREADS and PHONON_WIN_CORRECTION_THREADS override only the
//! explicit thread flags; unset keeps the existing CPU-count defaults.
//! PHONON_WIN_LLAMA_SERVER must name an absolute llama-server.exe from an
//! independently verified official release matching manifest::LLAMA_BUILD,
//! with its matching DLLs beside it. This opt-in executes trusted local code;
//! the version check is not an authenticity or archive-integrity check.
//!
//! PHONON_WIN_TRIM_IDLE defaults to 1; 0 skips best-effort working-set trims
//! after warm-up and delivered corrections, without unloading the model.
//!
//! All other supported llama.cpp environment controls remain inherited,
//! including LLAMA_ARG_N_PARALLEL, LLAMA_ARG_CACHE_RAM, LLAMA_ARG_BATCH,
//! LLAMA_ARG_UBATCH, LLAMA_ARG_LAZY_MODE and LLAMA_ARG_N_GPU_LAYERS.
//! --threads-batch has no environment form in b10726 and is not exposed. Context, model, loopback binding and sampling
//! remain owned by Phonon; there is deliberately no generic extra-args escape.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use crate::manifest;

const MAX_THREADS: usize = 256;
const SERVER_ENV: &str = "PHONON_WIN_LLAMA_SERVER";
const TRIM_IDLE_ENV: &str = "PHONON_WIN_TRIM_IDLE";

/// Windows trims only the owned correction child's working set, by default.
pub fn trim_idle_from_env() -> Result<bool> {
    parse_trim_idle(std::env::var_os(TRIM_IDLE_ENV).as_deref())
}

fn parse_trim_idle(value: Option<&OsStr>) -> Result<bool> {
    match value {
        None => Ok(true),
        Some(value) => match value.to_str() {
            Some("0") => Ok(false),
            Some("1") => Ok(true),
            _ => bail!("{TRIM_IDLE_ENV} must be 0 or 1 (unset defaults to 1)"),
        },
    }
}

pub fn threads_from_env(name: &str, default: usize) -> Result<usize> {
    parse_threads(name, std::env::var_os(name).as_deref(), default)
}

fn parse_threads(name: &str, value: Option<&OsStr>, default: usize) -> Result<usize> {
    let Some(value) = value else {
        return Ok(default);
    };
    let value = value
        .to_str()
        .with_context(|| format!("{name} must be a decimal integer in 1..={MAX_THREADS}"))?;
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("{name} must be a decimal integer in 1..={MAX_THREADS}");
    }
    let threads: usize = value
        .parse()
        .with_context(|| format!("{name} must be a decimal integer in 1..={MAX_THREADS}"))?;
    if !(1..=MAX_THREADS).contains(&threads) {
        bail!("{name} must be a decimal integer in 1..={MAX_THREADS}");
    }
    Ok(threads)
}

pub fn server_override() -> Result<Option<PathBuf>> {
    std::env::var_os(SERVER_ENV)
        .map(|path| validate_server_path(Path::new(&path)))
        .transpose()
}

fn validate_server_path(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        bail!("{SERVER_ENV} must be an absolute path to llama-server.exe");
    }
    if !path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("llama-server.exe"))
    {
        bail!("{SERVER_ENV} must name llama-server.exe, not a directory or another program");
    }
    if !path.is_file() {
        bail!(
            "{SERVER_ENV} does not name an existing file: {}",
            path.display()
        );
    }
    Ok(path.to_path_buf())
}

/// Official --version does not load a model. output() drains both streams,
/// unlike the resident server's formerly unconsumed pipes.
pub fn validate_server_build(server: &Path) -> Result<()> {
    let mut command = Command::new(server);
    command.arg("--version").stdin(Stdio::null());
    if let Some(parent) = server.parent() {
        command.current_dir(parent);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(crate::asr::CREATE_NO_WINDOW);
    }
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("check {SERVER_ENV} --version"))?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if started.elapsed() < Duration::from_secs(15) => {
                std::thread::sleep(Duration::from_millis(25));
            }
            state => {
                let _ = child.kill();
                let _ = child.wait();
                match state {
                    Err(error) => return Err(error).context("wait for correction runtime version"),
                    _ => bail!("{SERVER_ENV} --version did not exit within 15 seconds"),
                }
            }
        }
    }
    let output = child
        .wait_with_output()
        .context("read correction runtime version")?;
    let version = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() || !reported_build_matches(&version) {
        bail!(
            "{SERVER_ENV} must report pinned llama.cpp {} via --version; status {}",
            manifest::LLAMA_BUILD,
            output.status
        );
    }
    Ok(())
}

fn reported_build_matches(output: &str) -> bool {
    let expected = manifest::LLAMA_BUILD
        .strip_prefix('b')
        .unwrap_or(manifest::LLAMA_BUILD);
    output.lines().any(|line| {
        let Some(version) = line.trim().strip_prefix("version:") else {
            return false;
        };
        let legacy = version
            .split_whitespace()
            .next()
            .is_some_and(|build| build.strip_prefix('b').unwrap_or(build) == expected);
        let current = version
            .split_once("(build ")
            .and_then(|(_, build)| build.split_once(','))
            .is_some_and(|(build, _)| build.trim() == expected);
        legacy || current
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_trim_defaults_on_and_accepts_only_zero_or_one() {
        assert!(parse_trim_idle(None).unwrap());
        assert!(!parse_trim_idle(Some(OsStr::new("0"))).unwrap());
        assert!(parse_trim_idle(Some(OsStr::new("1"))).unwrap());
        for value in [
            "", "true", "false", "2", "-1", "+1", "00", " 0", "1 ", "0\n",
        ] {
            let error = parse_trim_idle(Some(OsStr::new(value))).unwrap_err();
            assert!(
                error.to_string().contains(TRIM_IDLE_ENV),
                "{value:?}: {error}"
            );
        }
    }

    #[test]
    fn idle_trim_rejects_non_unicode_values() {
        #[cfg(windows)]
        let value = {
            use std::os::windows::ffi::OsStringExt;
            std::ffi::OsString::from_wide(&[0xd800])
        };
        #[cfg(unix)]
        let value = {
            use std::os::unix::ffi::OsStringExt;
            std::ffi::OsString::from_vec(vec![0xff])
        };
        #[cfg(any(windows, unix))]
        assert!(parse_trim_idle(Some(&value)).is_err());
    }

    #[test]
    fn unset_threads_keep_existing_defaults() {
        assert_eq!(parse_threads("TEST", None, 11).unwrap(), 11);
        assert_eq!(parse_threads("TEST", None, 4).unwrap(), 4);
    }

    #[test]
    fn explicit_threads_are_positive_bounded_decimal_integers() {
        for (value, expected) in [("1", 1), ("4", 4), ("11", 11), ("256", 256)] {
            assert_eq!(
                parse_threads("TEST", Some(OsStr::new(value)), 11).unwrap(),
                expected
            );
        }
        for value in [
            "",
            "0",
            "-1",
            "+4",
            " 4",
            "4 ",
            "4.0",
            "four",
            "257",
            "184467440737095516160",
        ] {
            let error = parse_threads("TEST", Some(OsStr::new(value)), 11).unwrap_err();
            assert!(error.to_string().contains("TEST"), "{value}: {error}");
        }
    }

    #[test]
    fn external_server_path_must_be_absolute_and_an_existing_executable_file() {
        assert!(validate_server_path(Path::new("llama-server.exe")).is_err());
        assert!(validate_server_path(Path::new("")).is_err());
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "phonon-runtime-controls-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let server = root.join("llama-server.exe");
        assert!(validate_server_path(&server).is_err());
        std::fs::write(&server, b"validation fixture; never executed").unwrap();
        assert_eq!(validate_server_path(&server).unwrap(), server);
        let wrong_name = root.join("other.exe");
        std::fs::write(&wrong_name, b"never executed").unwrap();
        assert!(validate_server_path(&wrong_name).is_err());
        std::fs::remove_file(&wrong_name).unwrap();
        std::fs::remove_file(&server).unwrap();
        std::fs::create_dir(&server).unwrap();
        assert!(validate_server_path(&server).is_err());
        std::fs::remove_dir(&server).unwrap();
        std::fs::remove_dir(&root).unwrap();
    }

    #[test]
    fn external_build_must_report_the_exact_pin() {
        let number = manifest::LLAMA_BUILD.strip_prefix('b').unwrap();
        assert!(reported_build_matches(&format!(
            "version: {number} (abc123)\n"
        )));
        assert!(reported_build_matches(&format!(
            "backend diagnostic\nversion: {} (abc123)\nbuilt with MSVC\n",
            manifest::LLAMA_BUILD
        )));
        assert!(!reported_build_matches("version: 10725 (old)\n"));
        assert!(!reported_build_matches("version: 107260 (different)\n"));
        assert!(reported_build_matches(
            "version: 0.3.0-dev (build 10726, commit 85c55223c)\nbuilt with Clang 20.1.8 for Windows x86_64\n"
        ));
        assert!(!reported_build_matches(
            "version: 0.3.0-dev (build 107260, commit wrong)\n"
        ));
        assert!(!reported_build_matches(&format!("built with {number}")));
        assert!(!reported_build_matches("version: \n"));
        assert!(!reported_build_matches(""));
    }
}
