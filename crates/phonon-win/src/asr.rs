//! Speech recognition through sherpa-onnx.
//!
//! `sherpa-onnx-offline.exe` writes one JSON object per wave file to standard
//! output and everything else to standard error, so the parser only has to read
//! standard output. The model is the ONNX int8 export of Parakeet TDT 0.6b v2,
//! the same acoustic model the macOS build runs on MLX.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};

use crate::fetch;
use crate::manifest;

/// `CREATE_NO_WINDOW`: run a console child without a console window.
#[cfg(windows)]
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Explicit Windows opt-ins only. External executables and matching DLLs must
/// come from an independently verified official sherpa-onnx 1.13.6 archive.
#[cfg(any(windows, test))]
mod windows_runtime {
    use std::ffi::OsStr;
    use std::path::{Path, PathBuf};

    use anyhow::{bail, Result};

    const EXE_ENV: &str = "PHONON_WIN_ASR_EXE";
    const PROVIDER_ENV: &str = "PHONON_WIN_ASR_PROVIDER";

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum Provider {
        Cpu,
        Cuda,
    }

    impl Provider {
        pub(super) fn as_str(self) -> &'static str {
            match self {
                Self::Cpu => "cpu",
                Self::Cuda => "cuda",
            }
        }

        fn parse(value: Option<&OsStr>) -> Result<Self> {
            match value {
                None => Ok(Self::Cpu),
                Some(value) if value == OsStr::new("cpu") => Ok(Self::Cpu),
                Some(value) if value == OsStr::new("cuda") => Ok(Self::Cuda),
                _ => bail!("{PROVIDER_ENV} must be exactly cpu or cuda"),
            }
        }
    }

    pub(super) fn from_env(default: &Path) -> Result<(PathBuf, Provider)> {
        resolve(
            default,
            std::env::var_os(EXE_ENV).as_deref(),
            std::env::var_os(PROVIDER_ENV).as_deref(),
        )
    }

    fn resolve(
        default: &Path,
        executable: Option<&OsStr>,
        provider: Option<&OsStr>,
    ) -> Result<(PathBuf, Provider)> {
        let provider = Provider::parse(provider)?;
        let tool = match executable {
            None => super::absolute(default)?,
            Some(value) => {
                let path = Path::new(value);
                if !path.is_absolute() {
                    bail!("{EXE_ENV} must be an absolute path to sherpa-onnx-offline.exe");
                }
                if !path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.eq_ignore_ascii_case("sherpa-onnx-offline.exe"))
                    || !path.is_file()
                {
                    bail!(
                        "{EXE_ENV} must name an existing sherpa-onnx-offline.exe file: {}",
                        path.display()
                    );
                }
                path.to_path_buf()
            }
        };
        if provider == Provider::Cuda {
            if executable.is_none() {
                bail!("{PROVIDER_ENV}=cuda requires {EXE_ENV}; the bundled runtime is CPU-only");
            }
            let cuda = tool
                .parent()
                .expect("absolute executable has a parent")
                .join("onnxruntime_providers_cuda.dll");
            if !cuda.is_file() {
                bail!("CUDA requested but {} is missing; use the complete official CUDA runtime with CUDA 12/cuDNN 9 dependencies, not the CPU package", cuda.display());
            }
        }
        Ok((tool, provider))
    }

    pub(super) fn reject_provider_fallback(provider: Provider, stderr: &str) -> Result<()> {
        if provider == Provider::Cuda
            && stderr.lines().any(|line| {
                let line = line.to_ascii_lowercase();
                // This is the pinned session.cc CUDA-unavailable warning. Check
                // stderr only: a valid transcript may itself say "fallback to cpu".
                line.contains("available providers:") && line.contains("fallback to cpu")
            })
        {
            bail!("CUDA was requested but sherpa-onnx reported a CPU provider fallback; refusing the result without a CPU retry: {}", stderr.trim());
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        struct Fixture {
            dir: PathBuf,
        }

        impl Fixture {
            fn new() -> Self {
                let nonce = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let dir = std::env::temp_dir().join(format!(
                    "phonon asr controls {} {nonce}",
                    std::process::id()
                ));
                std::fs::create_dir(&dir).unwrap();
                Self { dir }
            }

            fn exe(&self) -> PathBuf {
                self.dir.join("sherpa-onnx-offline.exe")
            }

            fn write_exe(&self) {
                std::fs::write(self.exe(), b"validation fixture; never executed").unwrap();
            }
        }

        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(self.exe());
                let _ = std::fs::remove_file(self.dir.join("onnxruntime_providers_cuda.dll"));
                let _ = std::fs::remove_dir(&self.dir);
            }
        }

        #[test]
        fn provider_defaults_to_cpu_and_accepts_only_explicit_cpu_or_cuda() {
            assert_eq!(Provider::parse(None).unwrap(), Provider::Cpu);
            assert_eq!(
                Provider::parse(Some(OsStr::new("cpu"))).unwrap(),
                Provider::Cpu
            );
            assert_eq!(
                Provider::parse(Some(OsStr::new("cuda"))).unwrap(),
                Provider::Cuda
            );
            for value in [
                "",
                "auto",
                "CUDA",
                " cuda",
                "cuda ",
                "cuda:config.txt",
                "directml",
            ] {
                assert!(Provider::parse(Some(OsStr::new(value)))
                    .unwrap_err()
                    .to_string()
                    .contains(PROVIDER_ENV));
            }
        }

        #[test]
        fn unset_runtime_preserves_the_default_and_makes_it_absolute() {
            let default = Path::new("runtime/sherpa/bin/sherpa-onnx-offline.exe");
            let (tool, provider) = resolve(default, None, None).unwrap();
            assert_eq!(tool, super::super::absolute(default).unwrap());
            assert_eq!(provider, Provider::Cpu);
        }

        #[test]
        fn executable_override_rejects_relative_missing_directory_and_wrong_name_paths() {
            let fixture = Fixture::new();
            let default = fixture.exe();
            for path in [
                PathBuf::new(),
                PathBuf::from("sherpa-onnx-offline.exe"),
                fixture.dir.clone(),
                fixture.exe(),
                fixture.dir.join("other.exe"),
            ] {
                assert!(
                    resolve(&default, Some(path.as_os_str()), None).is_err(),
                    "{path:?}"
                );
            }
            fixture.write_exe();
            let (tool, provider) =
                resolve(&default, Some(fixture.exe().as_os_str()), None).unwrap();
            assert_eq!(tool, fixture.exe());
            assert_eq!(provider, Provider::Cpu);
        }

        #[test]
        fn cuda_requires_an_explicit_executable_and_its_provider_dll() {
            let fixture = Fixture::new();
            fixture.write_exe();
            let cuda = Some(OsStr::new("cuda"));
            assert!(resolve(&fixture.exe(), None, cuda)
                .unwrap_err()
                .to_string()
                .contains("CPU-only"));
            assert!(
                resolve(&fixture.exe(), Some(fixture.exe().as_os_str()), cuda)
                    .unwrap_err()
                    .to_string()
                    .contains("onnxruntime_providers_cuda.dll")
            );
            std::fs::write(
                fixture.dir.join("onnxruntime_providers_cuda.dll"),
                b"not loaded",
            )
            .unwrap();
            let (tool, provider) =
                resolve(&fixture.exe(), Some(fixture.exe().as_os_str()), cuda).unwrap();
            assert_eq!(tool, fixture.exe());
            assert_eq!(provider, Provider::Cuda);
        }

        #[test]
        fn a_successful_cpu_fallback_cannot_masquerade_as_cuda_success() {
            let warning = "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON. Available providers: CPUExecutionProvider, . Fallback to cpu!";
            assert!(reject_provider_fallback(Provider::Cuda, warning).is_err());
            assert!(
                reject_provider_fallback(Provider::Cuda, &warning.to_ascii_uppercase()).is_err()
            );
            assert!(reject_provider_fallback(Provider::Cpu, warning).is_ok());
            assert!(reject_provider_fallback(Provider::Cuda, "provider=\"cuda\"\nDone!").is_ok());
            assert!(reject_provider_fallback(Provider::Cuda, "").is_ok());
        }
    }
}

/// A ready recogniser: the tool and the weights it needs.
#[derive(Debug, Clone)]
pub struct Recognizer {
    tool: PathBuf,
    model_dir: PathBuf,
    threads: usize,
    #[cfg(windows)]
    provider: windows_runtime::Provider,
}

/// One transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
}

/// Pull the transcript out of one `sherpa-onnx-offline` standard-output block.
///
/// Kept separate from the process call so the parser is provable without Windows.
pub fn parse_output(stdout: &str) -> Result<Transcript> {
    for line in stdout.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(line).with_context(|| format!("parse sherpa result {line}"))?;
        let text = value
            .get("text")
            .and_then(|text| text.as_str())
            .ok_or_else(|| anyhow!("sherpa result has no text field: {line}"))?;
        return Ok(Transcript {
            text: text.trim().to_string(),
        });
    }
    bail!("sherpa-onnx-offline printed no result object")
}

/// The path, made absolute against the current directory. Nothing is resolved:
/// a symbolic link stays a symbolic link, and no extended-length prefix is added.
pub fn absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(std::env::current_dir()
        .context("read the current directory")?
        .join(path))
}

/// sherpa-onnx 1.13.6 aborts on a WAV with a valid header but no samples.
/// Treat that recording as silence before starting the native child.
fn wav_has_samples(path: &Path) -> Result<bool> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("read WAV header {}", path.display()))?;
    Ok(reader.duration() != 0)
}

/// How many decoding threads to use. sherpa-onnx gains little past four, and
/// leaving cores free keeps the machine responsive while the user dictates.
pub fn thread_count(logical_cpus: usize) -> usize {
    logical_cpus.saturating_sub(1).clamp(1, 4)
}

impl Recognizer {
    /// Locate an installed recogniser after the selected speech assets are
    /// installed.
    pub fn installed() -> Result<Self> {
        let tool = fetch::sentinel_path(&manifest::SHERPA_RUNTIME);
        #[cfg(windows)]
        let (tool, provider) = windows_runtime::from_env(&tool)?;
        if !tool.is_file() {
            bail!("{} is missing; run the first-run download", tool.display());
        }
        let model_dir = fetch::component_dir(&manifest::ASR_MODEL);
        #[cfg(windows)]
        let model_dir = absolute(&model_dir)?;
        if !model_dir.join("encoder.int8.onnx").is_file() {
            bail!("{} is missing the encoder", model_dir.display());
        }
        let threads = thread_count(
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(2),
        );
        #[cfg(windows)]
        let threads = crate::config::threads_from_env("PHONON_WIN_ASR_THREADS", threads)?;
        Ok(Self {
            tool,
            model_dir,
            threads,
            #[cfg(windows)]
            provider,
        })
    }

    /// The command line, so a failure can be reported exactly as it was run.
    pub fn args(&self, wav: &Path) -> Vec<String> {
        let mut args = vec![
            format!(
                "--encoder={}",
                self.model_dir.join("encoder.int8.onnx").display()
            ),
            format!(
                "--decoder={}",
                self.model_dir.join("decoder.int8.onnx").display()
            ),
            format!(
                "--joiner={}",
                self.model_dir.join("joiner.int8.onnx").display()
            ),
            format!("--tokens={}", self.model_dir.join("tokens.txt").display()),
            "--model-type=nemo_transducer".into(),
            format!("--num-threads={}", self.threads),
        ];
        #[cfg(windows)]
        args.push(format!("--provider={}", self.provider.as_str()));
        args.push(wav.display().to_string());
        args
    }

    /// Transcribe one 16-bit single-channel wave file.
    pub fn transcribe(&self, wav: &Path) -> Result<Transcript> {
        if !wav.is_file() {
            bail!("{} is not a file", wav.display());
        }
        // The tool runs from its own directory so it finds onnxruntime.dll, which
        // means a relative wave path would resolve against that directory
        // instead of the caller's. Make it absolute here. `canonicalize` would
        // do it too, but on Windows it returns an extended-length `\\?\` path
        // that not every C++ file reader accepts.
        let wav = absolute(wav)?;
        if !wav_has_samples(&wav)? {
            return Ok(Transcript {
                text: String::new(),
            });
        }
        let mut command = Command::new(&self.tool);
        #[cfg(windows)]
        {
            // A console tool started from a tray app gets its own console
            // window unless told otherwise; it flashed on the desktop on every
            // utterance on a real Windows Server 2022 box.
            use std::os::windows::process::CommandExt;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        #[cfg(target_os = "linux")]
        crate::linux::runtime_libraries(
            &mut command,
            &fetch::component_dir(&manifest::SHERPA_RUNTIME),
        )?;
        command.args(self.args(&wav));
        if let Some(bin) = self.tool.parent() {
            command.current_dir(bin);
        }
        let output = command
            .output()
            .with_context(|| format!("run {}", self.tool.display()))?;
        #[cfg(windows)]
        if !output.status.success() && self.provider == windows_runtime::Provider::Cuda {
            bail!(
                "{} (provider cuda) exited with {}: {}. No CPU retry; verify matching runtime DLLs, CUDA 12, cuDNN 9, and driver availability",
                self.tool.display(),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        if !output.status.success() {
            bail!(
                "sherpa-onnx-offline exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        #[cfg(windows)]
        windows_runtime::reject_provider_fallback(
            self.provider,
            &String::from_utf8_lossy(&output.stderr),
        )?;
        parse_output(&String::from_utf8_lossy(&output.stdout))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One real standard-output block, copied from the sherpa-onnx manual.
    const SAMPLE: &str = r#"{"lang": "", "emotion": "", "event": "", "text": " Well, I don't wish to see it any more, observed Phebe.", "timestamps": [0.32, 0.64], "tokens":[" Well", ","], "words": []}"#;

    #[test]
    fn reads_the_text_field() {
        let parsed = parse_output(SAMPLE).unwrap();
        assert_eq!(
            parsed.text,
            "Well, I don't wish to see it any more, observed Phebe."
        );
    }

    /// Standard error is redirected into the same stream by some shells. The
    /// parser must skip the noise and find the object.
    #[test]
    fn skips_lines_that_are_not_the_result() {
        let mixed =
            format!("Creating recognizer ...\nStarted\nDone!\n\n./foo.wav\n{SAMPLE}\n----\n");
        assert!(parse_output(&mixed).unwrap().text.starts_with("Well"));
    }

    #[test]
    fn an_empty_transcript_is_still_a_result() {
        let empty = r#"{"text": "", "tokens":[]}"#;
        assert_eq!(parse_output(empty).unwrap().text, "");
    }

    #[test]
    fn no_result_object_is_an_error() {
        assert!(parse_output("Creating recognizer ...\nStarted\n").is_err());
    }

    #[test]
    fn a_relative_path_becomes_absolute() {
        let made = absolute(Path::new("assets/startup.wav")).unwrap();
        assert!(made.is_absolute());
        assert!(made.ends_with("assets/startup.wav"));
        // An absolute path is handed back untouched, prefix and all.
        let already = std::env::temp_dir().join("x.wav");
        assert_eq!(absolute(&already).unwrap(), already);
    }

    #[test]
    fn empty_wavs_are_detected_before_starting_sherpa() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("empty.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        hound::WavWriter::create(&path, spec)
            .unwrap()
            .finalize()
            .unwrap();
        assert!(!wav_has_samples(&path).unwrap());
    }

    #[test]
    fn thread_count_leaves_a_core_free_and_stops_at_four() {
        assert_eq!(thread_count(1), 1);
        assert_eq!(thread_count(2), 1);
        assert_eq!(thread_count(4), 3);
        assert_eq!(thread_count(8), 4);
        assert_eq!(thread_count(32), 4);
    }

    #[test]
    fn runtime_arguments_keep_the_exact_int8_models_and_thread_count() {
        let recognizer = Recognizer {
            tool: PathBuf::from("unused"),
            model_dir: PathBuf::from("model with spaces"),
            threads: 4,
            #[cfg(windows)]
            provider: windows_runtime::Provider::Cuda,
        };
        let wav = Path::new("audio with spaces.wav");
        let args = recognizer.args(wav);
        for name in [
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ] {
            assert!(args.iter().any(|arg| arg.ends_with(name)), "{name}");
        }
        assert!(args.iter().any(|arg| arg == "--model-type=nemo_transducer"));
        assert!(args.iter().any(|arg| arg == "--num-threads=4"));
        assert_eq!(args.last().unwrap(), "audio with spaces.wav");
        #[cfg(windows)]
        assert!(args.iter().any(|arg| arg == "--provider=cuda"));
        #[cfg(not(windows))]
        assert!(!args.iter().any(|arg| arg.starts_with("--provider=")));
    }
}
