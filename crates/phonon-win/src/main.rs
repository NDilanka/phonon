//! Phonon for Windows.
//!
//! One executable. It carries the correction prompt and the startup fixture and
//! nothing else; the speech runtime, the correction runtime, and both models are
//! downloaded on first run and checked against pinned SHA-256 hashes.
//!
//! The crate builds on every platform so the portable half, the manifest, the
//! download and verify path, the output parsers, and the readiness rules, is
//! covered by the macOS test suite as well. Only `run` needs Windows.

// Half of this crate exists for the Windows application, which is not built on
// other platforms. Off Windows that half is unreachable, and saying so here is
// clearer than an attribute on every item.
#![cfg_attr(not(windows), allow(dead_code))]

mod asr;
#[cfg(any(windows, test))]
mod config;
mod dictionary;
mod fetch;
mod http;
mod manifest;
mod paths;
mod pipeline;
mod polish;
mod smoke;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(windows)]
mod win;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use pipeline::PassResult;

/// The startup fixture, shared with the macOS build. It is small enough to ship
/// inside the executable, and first run has to prove the audio path before
/// Phonon claims to be ready.
pub const STARTUP_WAV: &[u8] = include_bytes!("../../../assets/startup.wav");

#[derive(Parser)]
#[command(name = "phonon-win", version)]
#[cfg_attr(windows, command(about = "Phonon dictation for Windows"))]
#[cfg_attr(
    not(windows),
    command(about = "Phonon speech recognition and correction")
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the tray application. This is the default.
    Run,
    /// Download and verify the runtimes and models, then stop.
    Fetch,
    /// Print what first run downloads and where it goes.
    Info,
    /// Run one wave file through the real recogniser and the real correction
    /// model, then check the result. This is the release gate.
    Selftest {
        /// The wave file. Defaults to the fixture inside the executable.
        #[arg(long)]
        wav: Option<PathBuf>,
        /// Skip the fixture word check. Use for a file that is not the fixture.
        #[arg(long)]
        any_audio: bool,
    },
    /// Print local WAV transcripts and timings using one resident engine.
    /// Requires installed assets; never downloads. Output contains user text.
    Bench {
        /// WAV files, processed in the supplied order on every repeat.
        #[arg(required = true)]
        wav: Vec<PathBuf>,
        /// Repeat the sequence (1..=100, at most 256 passes in total).
        #[arg(long, default_value_t = 1, value_parser = parse_bench_repeat)]
        repeat: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Run) {
        Command::Run => run(),
        Command::Fetch => fetch_all(),
        Command::Info => {
            info();
            Ok(())
        }
        Command::Selftest { wav, any_audio } => selftest(wav, any_audio),
        Command::Bench { wav, repeat } => bench(wav, repeat),
    }
}

#[cfg(windows)]
fn run() -> Result<()> {
    win::run()
}

#[cfg(not(windows))]
fn run() -> Result<()> {
    bail!(
        "run is not supported on this platform; the tray, hotkey, and text insertion require Windows. \
         On Linux x86_64 use `phonon-win info`, `fetch`, or `selftest`. On macOS use `phonon`."
    )
}

fn info() {
    #[cfg(not(target_os = "linux"))]
    println!("Phonon for Windows");
    #[cfg(target_os = "linux")]
    println!("Phonon for Linux x86_64");
    println!("  data root:        {}", paths::data_root().display());
    let use_gmi = polish::uses_gmi().unwrap_or(false);
    let components = fetch::components_for_backend(use_gmi);
    println!(
        "  correction backend: {}",
        if use_gmi { "GMI Cloud" } else { "local Gemma" }
    );
    println!(
        "  first run downloads {}",
        manifest::human_bytes(
            components
                .iter()
                .map(|component| component.download_bytes())
                .sum()
        )
    );
    for component in &components {
        println!(
            "  {:<28} {:>9}  {}  {}",
            component.label,
            manifest::human_bytes(component.download_bytes()),
            if fetch::installed(component) {
                "installed"
            } else {
                "missing  "
            },
            component.dir
        );
    }
    println!(
        "  speech runtime:   sherpa-onnx {}",
        manifest::SHERPA_VERSION
    );
    println!("  correction runtime: llama.cpp {}", manifest::LLAMA_BUILD);
    println!(
        "  speech model:     {}@{}",
        manifest::ASR_MODEL_ID,
        &manifest::ASR_MODEL_REVISION[..7]
    );
    if use_gmi {
        println!("  GMI model:         {}", polish::configured_gmi_model());
    } else {
        println!(
            "  correction model: {}@{}",
            manifest::POLISH_MODEL_ID,
            &manifest::POLISH_MODEL_REVISION[..7]
        );
    }
}

fn fetch_all() -> Result<()> {
    use std::io::{IsTerminal, Write};
    // A terminal gets one line rewritten in place; a log gets one line per file and
    // per ten percent, instead of one per 4 MB chunk.
    let tty = std::io::stdout().is_terminal();
    let mut last = String::new();
    let mut last_step: (&str, u32) = ("", u32::MAX);
    let use_gmi = polish::uses_gmi()?;
    let components = fetch::components_for_backend(use_gmi);
    fetch::ensure_components(&components, &mut |progress| {
        let line = progress.line();
        if line == last {
            return;
        }
        last = line.clone();
        if tty {
            print!("\r\x1b[2K{line}");
            let _ = std::io::stdout().flush();
        } else {
            let step = (progress.file, progress.percent() / 10);
            if step != last_step {
                println!("{line}");
                last_step = step;
            }
        }
    })?;
    if tty && !last.is_empty() {
        println!();
    }
    println!(
        "active runtime and model assets are installed and verified ({})",
        if use_gmi {
            "GMI Cloud correction"
        } else {
            "local correction"
        }
    );
    Ok(())
}

/// The release gate. It downloads what is missing, starts the real correction
/// backend, transcribes real audio, corrects it, and applies the same two rules
/// the macOS readiness contract applies.
fn selftest(wav: Option<PathBuf>, any_audio: bool) -> Result<()> {
    fetch_all()?;
    let wav = match wav {
        Some(path) => path,
        None => {
            let path = paths::data_root().join("startup.wav");
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, STARTUP_WAV).context("write the startup fixture")?;
            path
        }
    };
    println!("selftest: {}", wav.display());

    let mut engine = pipeline::Engine::start()?;
    let result = engine.run(&wav)?;
    if engine.redacts_text() {
        println!("  transcript: [redacted; cloud correction enabled]");
    } else {
        println!("  raw:      {:?}", result.raw);
        println!("  polished: {:?}", result.polished);
    }
    println!(
        "  asr {:.0} ms, correction {:.0} ms",
        result.asr_ms, result.polish_ms
    );

    if any_audio {
        assert_usable(&result)?;
    } else {
        assert_startup(&result)?;
    }
    println!("selftest passed");
    engine.trim_after_pass(&result);
    Ok(())
}

const MAX_BENCH_REPEAT: usize = 100;
const MAX_BENCH_FILES: usize = 32;
const MAX_BENCH_PASSES: usize = 256;

fn parse_bench_repeat(value: &str) -> std::result::Result<usize, String> {
    let error = || format!("repeat must be an integer in 1..={MAX_BENCH_REPEAT}");
    let repeat = value.parse::<usize>().map_err(|_| error())?;
    if !(1..=MAX_BENCH_REPEAT).contains(&repeat) {
        return Err(error());
    }
    Ok(repeat)
}

/// Reject bad inputs before starting a model or processing any audio.
fn validate_bench_inputs(wavs: &[PathBuf], repeat: usize) -> Result<Vec<PathBuf>> {
    if wavs.is_empty() || wavs.len() > MAX_BENCH_FILES {
        bail!("bench requires 1..={MAX_BENCH_FILES} WAV paths");
    }
    if !(1..=MAX_BENCH_REPEAT).contains(&repeat) {
        bail!("repeat must be in 1..={MAX_BENCH_REPEAT}");
    }
    if repeat > MAX_BENCH_PASSES / wavs.len() {
        bail!("bench permits at most {MAX_BENCH_PASSES} total passes");
    }
    wavs.iter()
        .map(|wav| {
            if !wav.is_file() {
                bail!("{} is not a file", wav.display());
            }
            asr::absolute(wav)
        })
        .collect()
}

/// Explicit, local diagnostics: no fixture writes, downloads, or persistent logs.
fn bench(wavs: Vec<PathBuf>, repeat: usize) -> Result<()> {
    if polish::uses_gmi()? {
        bail!("bench is local-only; set PHONON_WIN_CORRECTION_BACKEND=local to run it");
    }
    let wavs = validate_bench_inputs(&wavs, repeat)?;
    println!("bench: local diagnostics; output includes raw and polished text");
    let started = Instant::now();
    let engine = pipeline::Engine::start()?;
    println!(
        "engine_start_ms: {:.0}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    bench_passes(&wavs, repeat, engine)
}

/// Separate timed work from idle maintenance without sharing mutable captures.
trait BenchRunner {
    fn run(&mut self, wav: &Path) -> Result<PassResult>;
    fn after_pass(&mut self, _result: &PassResult) {}
}

impl BenchRunner for pipeline::Engine {
    fn run(&mut self, wav: &Path) -> Result<PassResult> {
        pipeline::Engine::run(self, wav)
    }

    fn after_pass(&mut self, result: &PassResult) {
        self.trim_after_pass(result);
    }
}

// Existing model-free bench tests can keep supplying a simple run closure.
#[cfg(test)]
impl<F: FnMut(&Path) -> Result<PassResult>> BenchRunner for F {
    fn run(&mut self, wav: &Path) -> Result<PassResult> {
        self(wav)
    }
}

fn bench_passes(wavs: &[PathBuf], repeat: usize, mut runner: impl BenchRunner) -> Result<()> {
    let mut failed = 0;
    for round in 1..=repeat {
        for (index, wav) in wavs.iter().enumerate() {
            println!(
                "pass {}/{} (repeat {round}/{repeat}): {}",
                (round - 1) * wavs.len() + index + 1,
                repeat * wavs.len(),
                wav.display()
            );
            let started = Instant::now();
            let outcome = runner.run(wav);
            let total_ms = started.elapsed().as_secs_f64() * 1000.0;
            match &outcome {
                Ok(result) => {
                    println!("  raw:      {:?}", result.raw);
                    println!("  polished: {:?}", result.polished);
                    println!(
                        "  asr {:.0} ms, correction {:.0} ms",
                        result.asr_ms, result.polish_ms
                    );
                    // Check this request, not the startup sentence or a prior
                    // pass. Empty ASR skips correction and fails usability below.
                    match assert_usable(result) {
                        Ok(()) => println!("  usability: passed"),
                        Err(error) => {
                            failed += 1;
                            println!("  usability: FAILED: {error:#}");
                        }
                    }
                }
                Err(error) => {
                    failed += 1;
                    println!("  error: {error:#}");
                }
            }
            println!("  total {total_ms:.0} ms");
            // Outside the measured interval AND after printed pass completion.
            // No insertion in bench; exercise the same trim before the next run.
            if let Ok(result) = &outcome {
                runner.after_pass(result);
            }
        }
    }
    if failed != 0 {
        bail!("{failed} benchmark pass(es) failed; see per-pass diagnostics");
    }
    Ok(())
}

/// Both readiness rules, for the known fixture.
pub fn assert_startup(result: &PassResult) -> Result<()> {
    if !smoke::asr_smoke_passed(&result.raw) {
        bail!(
            "the recogniser did not produce the fixture sentence; got {:?}",
            result.raw
        );
    }
    assert_usable(result)
}

/// The correction rule alone, for audio whose words are not known ahead of time.
pub fn assert_usable(result: &PassResult) -> Result<()> {
    if result.raw.trim().is_empty() {
        bail!("the recogniser returned nothing");
    }
    if !smoke::llm_audio_smoke_passed(&result.raw, &result.polished) {
        bail!(
            "correction did not return a usable sentence: {:?} became {:?}",
            result.raw,
            result.polished
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixture_ships_inside_the_executable() {
        // A RIFF header and about one second of 16 kHz mono audio.
        assert_eq!(&STARTUP_WAV[..4], b"RIFF");
        assert_eq!(&STARTUP_WAV[8..12], b"WAVE");
        assert!(STARTUP_WAV.len() > 20_000, "the fixture looks truncated");
    }

    fn result(raw: &str, polished: &str) -> PassResult {
        PassResult {
            raw: raw.into(),
            polished: polished.into(),
            asr_ms: 1.0,
            polish_ms: 1.0,
        }
    }

    #[test]
    fn the_gate_accepts_a_good_pass() {
        assert!(assert_startup(&result("hello fluid voice", "Hello, Fluid Voice.")).is_ok());
    }

    #[test]
    fn the_gate_rejects_a_wrong_transcript() {
        let error = assert_startup(&result("goodbye", "Goodbye."))
            .unwrap_err()
            .to_string();
        assert!(error.contains("fixture sentence"));
    }

    #[test]
    fn the_gate_rejects_a_collapsed_correction() {
        let error = assert_startup(&result("hello fluid voice", "Hello."))
            .unwrap_err()
            .to_string();
        assert!(error.contains("usable sentence"));
    }

    /// Recognition without correction is not a supported mode, so an empty
    /// correction must fail even when the transcript was right.
    #[test]
    fn the_gate_rejects_recognition_alone() {
        assert!(assert_startup(&result("hello fluid voice", "")).is_err());
        assert!(assert_usable(&result("", "anything")).is_err());
    }

    #[test]
    fn bench_cli_requires_paths_and_bounds_repeats() {
        assert!(Cli::try_parse_from(["phonon-win", "bench"]).is_err());
        for invalid in ["0", "101", "-1", "many", "184467440737095516160"] {
            assert!(
                Cli::try_parse_from(["phonon-win", "bench", "--repeat", invalid, "one.wav"])
                    .is_err(),
                "{invalid}"
            );
        }
        let cli = Cli::try_parse_from(["phonon-win", "bench", "one.wav"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Bench { repeat: 1, .. })
        ));
        let cli =
            Cli::try_parse_from(["phonon-win", "bench", "--repeat", "2", "one.wav", "two.wav"])
                .unwrap();
        match cli.command.unwrap() {
            Command::Bench { wav, repeat } => {
                assert_eq!(
                    wav,
                    vec![PathBuf::from("one.wav"), PathBuf::from("two.wav")]
                );
                assert_eq!(repeat, 2);
            }
            _ => panic!("expected bench"),
        }
        assert_eq!(parse_bench_repeat("100").unwrap(), 100);
    }

    #[test]
    fn bench_preflight_bounds_files_passes_and_checks_every_path() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let fixture = root.join("../../assets/startup.wav");
        assert!(validate_bench_inputs(&[], 1).is_err());
        assert!(validate_bench_inputs(std::slice::from_ref(&fixture), 0).is_err());
        assert!(validate_bench_inputs(std::slice::from_ref(&fixture), 101).is_err());
        assert!(validate_bench_inputs(&vec![fixture.clone(); 33], 1).is_err());
        assert!(validate_bench_inputs(&vec![fixture.clone(); 32], 9).is_err());
        assert_eq!(
            validate_bench_inputs(&vec![fixture.clone(); 32], 8)
                .unwrap()
                .len(),
            32
        );
        assert!(validate_bench_inputs(std::slice::from_ref(&fixture), 100).is_ok());
        assert!(validate_bench_inputs(&[root], 1).is_err());
        assert!(validate_bench_inputs(&[fixture, PathBuf::new()], 1).is_err());
    }

    #[test]
    fn bench_runs_each_request_in_order_on_every_repeat() {
        let paths = vec![PathBuf::from("one.wav"), PathBuf::from("two.wav")];
        let mut seen = Vec::new();
        bench_passes(&paths, 2, |path: &Path| {
            seen.push(path.to_path_buf());
            Ok(result("hello fluid voice", "Hello, Fluid Voice."))
        })
        .unwrap();
        assert_eq!(seen, [paths.clone(), paths].concat());
    }

    #[test]
    fn bench_runs_idle_maintenance_only_after_completed_successful_requests() {
        struct Runner<'a> {
            events: &'a mut Vec<&'static str>,
            calls: usize,
        }
        impl BenchRunner for Runner<'_> {
            fn run(&mut self, _: &Path) -> Result<PassResult> {
                self.events.push("run");
                self.calls += 1;
                if self.calls == 2 {
                    bail!("synthetic transcription error");
                }
                Ok(result("hello fluid voice", "Hello, Fluid Voice."))
            }

            fn after_pass(&mut self, pass: &PassResult) {
                assert_eq!(pass.polished, "Hello, Fluid Voice.");
                self.events.push("after");
            }
        }

        let mut events = Vec::new();
        let runner = Runner {
            events: &mut events,
            calls: 0,
        };
        let error = bench_passes(&[PathBuf::from("one.wav")], 3, runner).unwrap_err();
        assert!(error.to_string().contains("1 benchmark pass(es) failed"));
        assert_eq!(events, ["run", "after", "run", "run", "after"]);
    }

    #[test]
    fn bench_reports_request_failures_but_still_visits_later_inputs() {
        let paths = vec![PathBuf::from("one.wav"), PathBuf::from("two.wav")];
        let mut calls = 0;
        let error = bench_passes(&paths, 2, |_: &Path| {
            calls += 1;
            match calls {
                1 => bail!("synthetic transcription error"),
                2 => Ok(result("", "A hallucination.")),
                _ => Ok(result("hello fluid voice", "Hello, Fluid Voice.")),
            }
        })
        .unwrap_err();
        assert_eq!(calls, 4);
        assert!(error.to_string().contains("2 benchmark pass(es) failed"));
    }
}
