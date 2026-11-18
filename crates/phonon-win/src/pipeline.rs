//! One dictation pass: wave file in, finished text out.
//!
//! The recogniser runs per pass. The correction model stays resident, because
//! loading three gigabytes of weights for every utterance would dominate the
//! latency a user feels.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::asr::Recognizer;
use crate::polish::Corrector;

/// What one pass produced, and what each stage cost.
#[derive(Debug, Clone)]
pub struct PassResult {
    pub raw: String,
    pub polished: String,
    pub asr_ms: f64,
    pub polish_ms: f64,
}

impl PassResult {
    /// One line for the log.
    pub fn summary(&self) -> String {
        format!(
            "asr {:.0} ms -> {:?}; correction {:.0} ms -> {:?}",
            self.asr_ms, self.raw, self.polish_ms, self.polished
        )
    }

    /// Timing-only form used when transcript text must not enter logs.
    pub fn timing_summary(&self) -> String {
        format!(
            "asr {:.0} ms; correction {:.0} ms",
            self.asr_ms, self.polish_ms
        )
    }
}

/// The warm engine: a located recogniser and a running correction server.
pub struct Engine {
    recognizer: Recognizer,
    corrector: Corrector,
}

impl Engine {
    /// Locate the recogniser and start the correction server. Both must succeed;
    /// recognition without correction is not a supported mode.
    pub fn start() -> Result<Self> {
        let recognizer = Recognizer::installed().context("speech recognition is not installed")?;
        let corrector = Corrector::start().context("correction model did not start")?;
        Ok(Self {
            recognizer,
            corrector,
        })
    }

    /// The caller must finish insertion (or a diagnostic/warm-up pass) first.
    /// Keep this out of run(): trimming must not delay key-release to insertion.
    /// A mutable borrow serializes trim with run() and child shutdown, without
    /// a background task or a handle that could become stale. Silence is a no-op.
    pub fn trim_after_pass(&mut self, result: &PassResult) {
        #[cfg(windows)]
        if !result.polished.trim().is_empty() {
            self.corrector.trim_idle();
        }
        #[cfg(not(windows))]
        let _ = result;
    }

    pub fn redacts_text(&self) -> bool {
        self.corrector.redacts_text()
    }

    /// Run one wave file through both stages.
    pub fn run(&self, wav: &Path) -> Result<PassResult> {
        let started = Instant::now();
        let transcript = self
            .recognizer
            .transcribe(wav)
            .with_context(|| format!("transcribe {}", wav.display()))?;
        let asr_ms = started.elapsed().as_secs_f64() * 1000.0;

        finish_pass(transcript.text, asr_ms, |text| self.corrector.correct(text))
    }
}

fn finish_pass(
    raw: String,
    asr_ms: f64,
    correct: impl FnOnce(&str) -> Result<String>,
) -> Result<PassResult> {
    let (polished, polish_ms) = if raw.trim().is_empty() {
        (String::new(), 0.0)
    } else {
        let started = Instant::now();
        let polished = correct(&raw).context("correct the transcript")?;
        (polished, started.elapsed().as_secs_f64() * 1000.0)
    };

    Ok(PassResult {
        raw,
        polished,
        asr_ms,
        polish_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_whitespace_transcripts_skip_correction() {
        for raw in ["", " ", "\t\r\n", "\u{00a0}\u{2003}"] {
            let result = finish_pass(raw.into(), 412.4, |_| {
                panic!("correction must not run for an empty transcript")
            })
            .unwrap();
            assert_eq!(result.raw, raw);
            assert_eq!(result.polished, "");
            assert_eq!(result.asr_ms, 412.4);
            assert_eq!(result.polish_ms, 0.0);
        }
    }

    #[test]
    fn nonempty_transcripts_are_corrected_once_without_trimming() {
        let raw = " \tHello, 世界!\r\n";
        let mut seen = Vec::new();
        let result = finish_pass(raw.into(), 412.4, |text| {
            seen.push(text.to_owned());
            Ok("Hello, 世界!".into())
        })
        .unwrap();
        assert_eq!(seen, vec![raw.to_owned()]);
        assert_eq!(result.raw, raw);
        assert_eq!(result.polished, "Hello, 世界!");
        assert_eq!(result.asr_ms, 412.4);
    }

    #[test]
    fn the_summary_names_both_stages() {
        let result = PassResult {
            raw: "hello fluid voice".into(),
            polished: "Hello, Fluid Voice.".into(),
            asr_ms: 412.4,
            polish_ms: 980.6,
        };
        let line = result.summary();
        assert!(line.contains("asr 412 ms"));
        assert!(line.contains("correction 981 ms"));
        assert!(line.contains("Hello, Fluid Voice."));
    }
}
