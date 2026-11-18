//! Transcript correction through llama.cpp.
//!
//! The macOS build runs Gemma 4 E2B on MLX inside `sidecar/polish_server.py`.
//! Windows runs the same model, Google's quantisation-aware q4_0 GGUF, inside a
//! resident `llama-server.exe`. The prompt, the temperature, and the output
//! budget rule are the same, so both platforms correct a transcript the same way.
//!
//! Correction is mandatory. A recogniser result that never reaches this stage is
//! a failure, not a degraded mode.

#[cfg(not(any(windows, target_os = "linux")))]
use std::io::Read;
use std::path::PathBuf;
#[cfg(not(windows))]
use std::process::Child;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use phonon_core::data::DictionaryFile;

use crate::fetch;
use crate::http;
use crate::manifest;

const CORRECTION_BACKEND_ENV: &str = "PHONON_WIN_CORRECTION_BACKEND";
const GMI_API_KEY_ENV: &str = "GMI_API_KEY";
const GMI_MODEL_ENV: &str = "PHONON_WIN_GMI_MODEL";
const GMI_MODEL: &str = "google/gemini-3.8-flash";
const GMI_CHAT_URL: &str = "https://api.gmi-serving.com/v1/chat/completions";
const GMI_TIMEOUT: Duration = Duration::from_secs(15);
const GMI_MAX_TOKENS: u32 = 512;
const GMI_CHUNK_WORDS: usize = 64;
const GMI_MAX_RESPONSE_BYTES: u64 = 64 * 1024;

#[derive(Debug)]
struct GmiTruncated;

impl std::fmt::Display for GmiTruncated {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("GMI correction output was truncated")
    }
}

impl std::error::Error for GmiTruncated {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionBackend {
    Local,
    Gmi,
}

fn parse_backend(value: Option<&str>, has_key: bool) -> Result<CorrectionBackend> {
    let value = value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase);
    match value.as_deref() {
        Some("local") => Ok(CorrectionBackend::Local),
        Some("gmi") | Some("cloud") => Ok(CorrectionBackend::Gmi),
        Some(other) => bail!("{CORRECTION_BACKEND_ENV} must be local or gmi, got {other:?}"),
        None if has_key => Ok(CorrectionBackend::Gmi),
        None => Ok(CorrectionBackend::Local),
    }
}

/// Select cloud correction automatically when the approved API key is present.
/// Local remains available for offline use and regression diagnostics.
pub fn backend() -> Result<CorrectionBackend> {
    parse_backend(
        std::env::var(CORRECTION_BACKEND_ENV).ok().as_deref(),
        std::env::var_os(GMI_API_KEY_ENV).is_some(),
    )
}

pub fn uses_gmi() -> Result<bool> {
    Ok(backend()? == CorrectionBackend::Gmi)
}

pub fn configured_gmi_model() -> String {
    std::env::var(GMI_MODEL_ENV).unwrap_or_else(|_| GMI_MODEL.to_string())
}

/// The correction prompt, shared with the macOS sidecar. Embedded so the shipped
/// executable carries no loose files.
pub const SYSTEM_PROMPT: &str = include_str!("../../../prompts/polish_v2.txt");

/// Bounds on generated tokens, copied from `sidecar/polish_server.py`.
pub const OUTPUT_TOKEN_CEILING: u32 = 2048;
pub const OUTPUT_TOKEN_FLOOR: u32 = 48;

/// The model may take this long to load before the server is called dead.
const READY_TIMEOUT: Duration = Duration::from_secs(600);

/// Wrap a transcript the way the prompt expects.
pub fn user_message(text: &str) -> String {
    format!("<transcript>\n{}\n</transcript>", text.trim())
}

/// Keep the empty-dictionary request byte-identical to the original wrapper.
pub fn dictionary_user_message(dictionary: &DictionaryFile, text: &str) -> String {
    if dictionary.entries.is_empty() {
        user_message(text)
    } else {
        dictionary.prepare_polish_input(text)
    }
}

/// The text inside `<transcript>`, or the whole string when the tag is absent.
/// Mirrors `transcript_payload` in the macOS sidecar.
pub fn transcript_payload(text: &str) -> &str {
    let Some(open) = text.find("<transcript>") else {
        return text.trim();
    };
    let rest = &text[open + "<transcript>".len()..];
    match rest.find("</transcript>") {
        Some(close) => rest[..close].trim(),
        None => rest.trim(),
    }
}

/// Strip channel and turn markers a chat template can leak into the answer.
/// Mirrors `clean_output` in the macOS sidecar, whose pattern is
/// `<\|?/?(?:channel|turn|think)\|?>?`.
pub fn clean_output(text: &str) -> String {
    const MARKERS: [&str; 3] = ["channel", "turn", "think"];
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '<' {
            if let Some(width) = marker_width(&chars[index..], &MARKERS) {
                index += width;
                continue;
            }
        }
        out.push(chars[index]);
        index += 1;
    }
    out.trim().to_string()
}

/// How many characters the marker at the start of `rest` spans, if it is one.
fn marker_width(rest: &[char], markers: &[&str]) -> Option<usize> {
    let mut at = 1; // the '<' itself
    let eat = |at: &mut usize, wanted: char| {
        if rest.get(*at) == Some(&wanted) {
            *at += 1;
        }
    };
    eat(&mut at, '|');
    eat(&mut at, '/');
    let word = markers.iter().find(|marker| {
        marker
            .chars()
            .enumerate()
            .all(|(offset, wanted)| rest.get(at + offset) == Some(&wanted))
    })?;
    at += word.chars().count();
    eat(&mut at, '|');
    eat(&mut at, '>');
    Some(at)
}

/// How many tokens the correction may generate, given the spoken length.
/// A clean-up pass returns roughly its input, with headroom for expanded
/// orthography. Same rule as the macOS sidecar.
pub fn output_budget(spoken_tokens: u32) -> u32 {
    let budget = (spoken_tokens as f64 * 1.8).ceil() as u32 + 24;
    budget.clamp(OUTPUT_TOKEN_FLOOR, OUTPUT_TOKEN_CEILING)
}

/// A rough token count for text, used only when the server cannot tokenise.
/// English averages a little under four characters per token.
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.trim().chars().count() as u32;
    chars.div_ceil(4).max(1)
}

/// How many threads llama.cpp should use. One core stays free for the tray, the
/// hook, and whatever the user is typing into.
pub fn thread_count(logical_cpus: usize) -> usize {
    logical_cpus.saturating_sub(1).max(1)
}

/// A free loopback port. The server binds it a moment later; nothing else on the
/// machine is looking for a port in that instant.
fn free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").context("reserve a local port")?;
    Ok(listener.local_addr()?.port())
}

/// The answer inside one chat-completion reply.
///
/// A reply whose `content` is empty but whose `reasoning_content` is not means
/// the model reasoned instead of answering. Say so: it is a different fault
/// from a server that returned nothing at all.
pub fn parse_reply(value: &serde_json::Value) -> Result<String> {
    let choice = value
        .get("choices")
        .and_then(|choices| choices.get(0))
        .ok_or_else(|| anyhow!("correction reply has no choices: {value}"))?;
    let message = choice
        .get("message")
        .ok_or_else(|| anyhow!("correction reply has no message: {value}"))?;
    let content = message
        .get("content")
        .and_then(|content| content.as_str())
        .unwrap_or_default();
    if !content.trim().is_empty() {
        return Ok(content.to_string());
    }
    let reasoning = message
        .get("reasoning_content")
        .and_then(|reasoning| reasoning.as_str())
        .unwrap_or_default();
    let finish = choice
        .get("finish_reason")
        .and_then(|reason| reason.as_str())
        .unwrap_or("unknown");
    if reasoning.trim().is_empty() {
        bail!("correction returned an empty answer (finish_reason {finish})");
    }
    bail!(
        "correction reasoned instead of answering (finish_reason {finish}): {}",
        reasoning.trim()
    )
}

/// Best-effort trim control, independent of Win32 so the policy is unit tested.
#[cfg(any(windows, test))]
mod idle_trim {
    pub(super) struct IdleTrim {
        enabled: bool,
        failure_reported: bool,
    }

    impl IdleTrim {
        pub(super) fn new(enabled: bool) -> Self {
            Self {
                enabled,
                failure_reported: false,
            }
        }

        /// Retry each idle opportunity, but return only the first failure to log.
        pub(super) fn attempt(
            &mut self,
            trim: impl FnOnce() -> std::io::Result<()>,
        ) -> Option<std::io::Error> {
            if !self.enabled {
                return None;
            }
            match trim() {
                Err(error) if !self.failure_reported => {
                    self.failure_reported = true;
                    Some(error)
                }
                _ => None,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::io;

        #[test]
        fn disabled_trim_never_calls_the_native_operation() {
            let mut trim = IdleTrim::new(false);
            for _ in 0..3 {
                assert!(trim.attempt(|| panic!("trim is disabled")).is_none());
            }
        }

        #[test]
        fn enabled_trim_runs_on_every_attempt() {
            let mut trim = IdleTrim::new(true);
            let mut calls = 0;
            for _ in 0..3 {
                assert!(trim
                    .attempt(|| {
                        calls += 1;
                        Ok(())
                    })
                    .is_none());
            }
            assert_eq!(calls, 3);
        }

        #[test]
        fn failures_are_reported_once_but_do_not_stop_later_attempts() {
            let mut trim = IdleTrim::new(true);
            let mut calls = 0;
            for (fails, reports) in [(true, true), (true, false), (false, false), (true, false)] {
                let error = trim.attempt(|| {
                    calls += 1;
                    if fails {
                        Err(io::Error::from(io::ErrorKind::PermissionDenied))
                    } else {
                        Ok(())
                    }
                });
                assert_eq!(error.is_some(), reports);
            }
            assert_eq!(calls, 4);
        }
    }
}

#[cfg(any(windows, test))]
mod startup_diagnostics {
    use std::collections::VecDeque;
    use std::io::{self, Read};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, JoinHandle};

    const TAIL_BYTES: usize = 64 * 1024;
    const READ_BYTES: usize = 4096;

    fn append_tail(tail: &mut VecDeque<u8>, bytes: &[u8]) {
        if bytes.len() >= TAIL_BYTES {
            tail.clear();
            tail.extend(&bytes[bytes.len() - TAIL_BYTES..]);
        } else {
            let excess = (tail.len() + bytes.len()).saturating_sub(TAIL_BYTES);
            tail.drain(..excess);
            tail.extend(bytes);
        }
    }

    struct Shared {
        // None after readiness: discard startup bytes and retain no later text.
        tail: Mutex<Option<VecDeque<u8>>>,
        stopping: AtomicBool,
        read_failed: AtomicBool,
    }

    impl Shared {
        fn record(&self, label: &str, bytes: &[u8]) {
            let mut guard = self.tail.lock().unwrap_or_else(|error| error.into_inner());
            if let Some(tail) = guard.as_mut() {
                append_tail(tail, b"\n[");
                append_tail(tail, label.as_bytes());
                append_tail(tail, b"] ");
                append_tail(tail, bytes);
            }
        }

        fn bytes(&self) -> Vec<u8> {
            self.tail
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_ref()
                .map(|tail| tail.iter().copied().collect())
                .unwrap_or_default()
        }
    }

    fn drain(mut reader: impl Read, shared: &Shared, label: &str) {
        let mut buffer = [0; READ_BYTES];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => return,
                Ok(count) => shared.record(label, &buffer[..count]),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    shared.record(label, format!("pipe read failed: {error}").as_bytes());
                    shared.read_failed.store(true, Ordering::Release);
                    return;
                }
            }
        }
    }

    struct Diagnostics {
        shared: Arc<Shared>,
        threads: Vec<(&'static str, JoinHandle<()>)>,
    }

    impl Diagnostics {
        fn new() -> Self {
            Self {
                shared: Arc::new(Shared {
                    tail: Mutex::new(Some(VecDeque::with_capacity(TAIL_BYTES))),
                    stopping: AtomicBool::new(false),
                    read_failed: AtomicBool::new(false),
                }),
                threads: Vec::new(),
            }
        }

        fn start_reader(
            &mut self,
            label: &'static str,
            reader: impl Read + Send + 'static,
        ) -> io::Result<()> {
            let shared = Arc::clone(&self.shared);
            let handle = thread::Builder::new()
                .name(format!("phonon-llama-{label}"))
                .spawn(move || drain(reader, &shared, label))?;
            self.threads.push((label, handle));
            Ok(())
        }

        fn ready(&self) {
            *self
                .shared
                .tail
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = None;
        }

        fn finish(&mut self) {
            self.shared.stopping.store(true, Ordering::Release);
            for (label, handle) in self.threads.drain(..) {
                if handle.join().is_err() {
                    self.shared.record(label, b"diagnostic reader panicked");
                }
            }
        }
    }

    impl Drop for Diagnostics {
        fn drop(&mut self) {
            self.finish();
        }
    }

    // std child pipes are synchronous on Windows. Peek before reading so a
    // descendant retaining a write handle cannot strand a reader at shutdown.
    // This is the Win32 ABI; no additional windows-sys feature is required.
    #[cfg(windows)]
    #[link(name = "kernel32")]
    extern "system" {
        #[link_name = "PeekNamedPipe"]
        fn peek_named_pipe(
            pipe: *mut std::ffi::c_void,
            buffer: *mut std::ffi::c_void,
            buffer_size: u32,
            bytes_read: *mut u32,
            available: *mut u32,
            bytes_left: *mut u32,
        ) -> i32;
    }

    #[cfg(windows)]
    struct PipeReader<R> {
        pipe: R,
        shared: Arc<Shared>,
        final_bytes: usize,
    }

    #[cfg(windows)]
    impl<R: Read + std::os::windows::io::AsRawHandle> Read for PipeReader<R> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if buffer.is_empty() {
                return Ok(0);
            }
            loop {
                let stopping = self.shared.stopping.load(Ordering::Acquire);
                if stopping && self.final_bytes == 0 {
                    return Ok(0);
                }
                let mut available = 0;
                // SAFETY: this thread owns the live read handle; PeekNamedPipe
                // only writes to `available`, and all other output pointers are null.
                let ok = unsafe {
                    peek_named_pipe(
                        self.pipe.as_raw_handle(),
                        std::ptr::null_mut(),
                        0,
                        std::ptr::null_mut(),
                        &mut available,
                        std::ptr::null_mut(),
                    )
                };
                if ok == 0 {
                    let error = io::Error::last_os_error();
                    return if matches!(error.raw_os_error(), Some(109 | 232)) {
                        Ok(0)
                    } else {
                        Err(error)
                    };
                }
                if available != 0 {
                    // Only one reader owns the handle, so already-buffered
                    // bytes cannot be consumed between the peek and the read.
                    let mut count = buffer.len().min(available as usize);
                    if stopping {
                        count = count.min(self.final_bytes);
                    }
                    let count = self.pipe.read(&mut buffer[..count])?;
                    if stopping {
                        self.final_bytes -= count;
                    }
                    return Ok(count);
                }
                if stopping {
                    return Ok(0);
                }
                thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }

    #[cfg(windows)]
    pub(super) struct Server {
        child: std::process::Child,
        diagnostics: Diagnostics,
        stopped: bool,
    }

    #[cfg(windows)]
    impl Server {
        pub(super) fn new(child: std::process::Child) -> anyhow::Result<Self> {
            use anyhow::Context;
            let mut server = Self {
                child,
                diagnostics: Diagnostics::new(),
                stopped: false,
            };
            let setup: anyhow::Result<()> = (|| {
                let stdout = server
                    .child
                    .stdout
                    .take()
                    .context("missing server stdout pipe")?;
                let stderr = server
                    .child
                    .stderr
                    .take()
                    .context("missing server stderr pipe")?;
                server.start_pipe("stdout", stdout)?;
                server.start_pipe("stderr", stderr)?;
                Ok(())
            })();
            if let Err(error) = setup {
                return Err(server.failure(error));
            }
            Ok(server)
        }

        fn start_pipe(
            &mut self,
            label: &'static str,
            pipe: impl Read + std::os::windows::io::AsRawHandle + Send + 'static,
        ) -> anyhow::Result<()> {
            use anyhow::Context;
            let reader = PipeReader {
                pipe,
                shared: Arc::clone(&self.diagnostics.shared),
                final_bytes: TAIL_BYTES,
            };
            self.diagnostics
                .start_reader(label, reader)
                .with_context(|| format!("start server {label} drain"))
        }

        pub(super) fn wait_until_ready(&mut self, base: &str) -> anyhow::Result<()> {
            let started = super::Instant::now();
            loop {
                match self.child.try_wait() {
                    Ok(Some(status)) => {
                        return Err(self.failure(anyhow::anyhow!(
                            "llama-server exited with {status} before it was ready"
                        )));
                    }
                    Ok(None) => {}
                    Err(error) => {
                        return Err(self.failure(anyhow::anyhow!(
                            "poll llama-server during startup: {error}"
                        )));
                    }
                }
                if self.diagnostics.shared.read_failed.load(Ordering::Acquire) {
                    return Err(self.failure(anyhow::anyhow!(
                        "llama-server startup diagnostic pipe failed"
                    )));
                }
                let health_error = match super::http::agent()
                    .get(&format!("{base}/health"))
                    .timeout(super::Duration::from_secs(5))
                    .call()
                {
                    Ok(_) => {
                        // Before Engine::start can return or accept user audio.
                        self.diagnostics.ready();
                        return Ok(());
                    }
                    Err(error) => error,
                };
                // Connection failures and HTTP 503 are normal while loading.
                // Retry health errors, but retain the final cause on timeout.
                if started.elapsed() > super::READY_TIMEOUT {
                    return Err(self.failure(anyhow::anyhow!(
                        "llama-server did not become ready within {:?}; last health error: {health_error}",
                        super::READY_TIMEOUT
                    )));
                }
                thread::sleep(super::Duration::from_millis(500));
            }
        }

        fn failure(&mut self, reason: anyhow::Error) -> anyhow::Error {
            self.stop();
            let bytes = self.diagnostics.shared.bytes();
            anyhow::anyhow!(
                "{reason:#}\nllama-server startup output (combined tail, {} bytes):\n{}",
                bytes.len(),
                String::from_utf8_lossy(&bytes)
            )
        }

        pub(super) fn trim_working_set(&mut self) -> std::io::Result<()> {
            use std::os::windows::io::AsRawHandle;

            if self.stopped {
                return Ok(());
            }
            // SAFETY: the actual Child retains this process handle for the
            // whole synchronous call. No PID lookup, copied handle or task can
            // outlive it; the exclusive borrow also excludes stop/drop.
            let ok = unsafe {
                windows_sys::Win32::System::ProcessStatus::EmptyWorkingSet(
                    self.child.as_raw_handle(),
                )
            };
            if ok == 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        }

        pub(super) fn stop(&mut self) {
            if !self.stopped {
                self.stopped = true;
                let _ = self.child.kill();
                let _ = self.child.wait();
                // Flush bounded queued output, then join even if an inherited
                // writer never closes its handle.
                self.diagnostics.finish();
            }
        }
    }

    #[cfg(windows)]
    impl Drop for Server {
        fn drop(&mut self) {
            self.stop();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::io::Cursor;
        use std::sync::atomic::AtomicUsize;
        use std::sync::mpsc::{self, Receiver};
        use std::time::Duration;

        struct Counted<R> {
            reader: R,
            bytes: Arc<AtomicUsize>,
        }

        impl<R: Read> Read for Counted<R> {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                let count = self.reader.read(buffer)?;
                self.bytes.fetch_add(count, Ordering::Relaxed);
                Ok(count)
            }
        }

        struct ChannelReader {
            receiver: Receiver<Vec<u8>>,
            pending: Cursor<Vec<u8>>,
        }

        impl Read for ChannelReader {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                loop {
                    let count = self.pending.read(buffer)?;
                    if count != 0 {
                        return Ok(count);
                    }
                    match self.receiver.recv_timeout(Duration::from_secs(2)) {
                        Ok(bytes) => self.pending = Cursor::new(bytes),
                        Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(0),
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            return Err(io::ErrorKind::TimedOut.into());
                        }
                    }
                }
            }
        }

        #[test]
        fn tail_is_bounded_for_small_and_oversized_writes() {
            let mut tail = VecDeque::with_capacity(TAIL_BYTES);
            append_tail(&mut tail, &vec![b'a'; TAIL_BYTES - 2]);
            append_tail(&mut tail, b"bcdef");
            assert_eq!(tail.len(), TAIL_BYTES);
            assert!(tail.iter().copied().collect::<Vec<_>>().ends_with(b"bcdef"));
            append_tail(&mut tail, &vec![b'z'; TAIL_BYTES * 4]);
            assert_eq!(tail.len(), TAIL_BYTES);
            assert!(tail.iter().all(|byte| *byte == b'z'));
        }

        #[test]
        fn long_unbroken_stream_never_requires_a_line_buffer() {
            let mut diagnostics = Diagnostics::new();
            let mut bytes = vec![b'x'; TAIL_BYTES * 8];
            bytes.extend_from_slice(b"precise final startup failure");
            diagnostics
                .start_reader("stderr", Cursor::new(bytes))
                .unwrap();
            diagnostics.finish();
            let tail = diagnostics.shared.bytes();
            assert_eq!(tail.len(), TAIL_BYTES);
            assert!(tail.ends_with(b"precise final startup failure"));
        }

        #[test]
        fn both_pipes_drain_concurrently_under_backpressure() {
            let mut diagnostics = Diagnostics::new();
            let (out, out_rx) = mpsc::sync_channel(1);
            let (err, err_rx) = mpsc::sync_channel(1);
            let mut counts = Vec::new();
            for (label, receiver) in [("stdout", out_rx), ("stderr", err_rx)] {
                let bytes = Arc::new(AtomicUsize::new(0));
                diagnostics
                    .start_reader(
                        label,
                        Counted {
                            reader: ChannelReader {
                                receiver,
                                pending: Cursor::new(Vec::new()),
                            },
                            bytes: Arc::clone(&bytes),
                        },
                    )
                    .unwrap();
                counts.push(bytes);
            }
            let (done, completed) = mpsc::channel();
            let writer = thread::spawn(move || {
                let ok = (0..256).all(|_| {
                    out.send(vec![b'o'; 1024]).is_ok() && err.send(vec![b'e'; 1024]).is_ok()
                });
                let _ = done.send(ok);
                ok
            });
            let outcome = completed.recv_timeout(Duration::from_secs(5));
            let writer_ok = writer.join().unwrap();
            diagnostics.finish();
            assert!(outcome.unwrap() && writer_ok, "a pipe was not drained");
            assert!(!diagnostics.shared.read_failed.load(Ordering::Acquire));
            for bytes in counts {
                assert_eq!(bytes.load(Ordering::Relaxed), 256 * 1024);
            }
            let tail = diagnostics.shared.bytes();
            assert!(tail.len() <= TAIL_BYTES);
            assert!(tail.windows(8).any(|bytes| bytes == b"[stdout]"));
            assert!(tail.windows(8).any(|bytes| bytes == b"[stderr]"));
        }

        #[test]
        fn readiness_clears_startup_bytes_and_keeps_draining_without_retention() {
            let mut diagnostics = Diagnostics::new();
            diagnostics.shared.record("stderr", b"startup only");
            assert!(!diagnostics.shared.bytes().is_empty());
            diagnostics.ready();
            assert!(diagnostics.shared.bytes().is_empty());
            let count = Arc::new(AtomicUsize::new(0));
            for label in ["stdout", "stderr"] {
                diagnostics
                    .start_reader(
                        label,
                        Counted {
                            reader: Cursor::new(vec![b'p'; TAIL_BYTES * 4]),
                            bytes: Arc::clone(&count),
                        },
                    )
                    .unwrap();
            }
            diagnostics.finish();
            assert_eq!(count.load(Ordering::Relaxed), TAIL_BYTES * 8);
            assert!(diagnostics.shared.bytes().is_empty());
        }
    }
}

/// The selected correction backend.
pub(crate) enum Corrector {
    Local(LocalCorrector),
    Gmi(GmiCorrector),
}

/// A running local `llama-server` with the correction model loaded.
pub(crate) struct LocalCorrector {
    #[cfg(not(windows))]
    child: Child,
    #[cfg(windows)]
    child: startup_diagnostics::Server,
    #[cfg(windows)]
    idle_trim: idle_trim::IdleTrim,
    base: String,
    system_prompt: String,
    dictionary: DictionaryFile,
}

/// A stateless HTTPS correction client. The API key is held only in memory.
pub(crate) struct GmiCorrector {
    api_key: String,
    model: String,
    dictionary: DictionaryFile,
}

impl LocalCorrector {
    /// Start the server and wait until the model answers. Call after
    /// the selected runtime and model assets are installed.
    fn start(dictionary: DictionaryFile) -> Result<Self> {
        // Reject malformed configuration before starting a child.
        #[cfg(windows)]
        let idle_trim = idle_trim::IdleTrim::new(crate::config::trim_idle_from_env()?);
        let server = fetch::sentinel_path(&manifest::LLAMA_RUNTIME);
        #[cfg(windows)]
        let server = match crate::config::server_override()? {
            Some(external) => {
                crate::config::validate_server_build(&external)?;
                external
            }
            None => server,
        };
        if !server.is_file() {
            bail!(
                "{} is missing; run the first-run download",
                server.display()
            );
        }
        let model: PathBuf =
            fetch::component_dir(&manifest::POLISH_MODEL).join(manifest::POLISH_GGUF);
        if !model.is_file() {
            bail!("{} is missing", model.display());
        }
        let port = free_port()?;
        let threads = thread_count(
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(2),
        );
        #[cfg(windows)]
        let threads = crate::config::threads_from_env("PHONON_WIN_CORRECTION_THREADS", threads)?;

        let mut command = Command::new(&server);
        command
            .arg("-m")
            .arg(&model)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-c")
            .arg("4096")
            .arg("-t")
            .arg(threads.to_string())
            .arg("--jinja")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = server.parent() {
            command.current_dir(dir);
        }
        #[cfg(windows)]
        {
            // Without this the server owns a console window that sits on the
            // desktop for the life of the tray app.
            use std::os::windows::process::CommandExt;
            command.creation_flags(crate::asr::CREATE_NO_WINDOW);
        }
        #[cfg(target_os = "linux")]
        let log_path = {
            crate::linux::runtime_libraries(
                &mut command,
                &fetch::component_dir(&manifest::LLAMA_RUNTIME),
            )?;
            // A pipe that nobody drains fills during model loading on Linux.
            // Keep diagnostics on disk so startup and later requests cannot block.
            let logs = crate::paths::data_root().join("logs");
            std::fs::create_dir_all(&logs)?;
            let path = logs.join(format!("llama-server-{port}.log"));
            let log = std::fs::File::create(&path).context("create the correction server log")?;
            command.stdout(Stdio::null()).stderr(log);
            command.arg("-ngl").arg("0");
            path
        };
        let child = command
            .spawn()
            .with_context(|| format!("start {}", server.display()))?;

        #[cfg(windows)]
        let mut child = startup_diagnostics::Server::new(child)?;
        #[cfg(not(windows))]
        let mut child = child;
        let base = format!("http://127.0.0.1:{port}");
        #[cfg(windows)]
        child.wait_until_ready(&base)?;
        #[cfg(not(windows))]
        {
            let started = Instant::now();
            loop {
                if let Some(status) = child.try_wait()? {
                    #[cfg(target_os = "linux")]
                    let stderr = std::fs::read_to_string(&log_path).unwrap_or_default();
                    #[cfg(not(any(windows, target_os = "linux")))]
                    let mut stderr = String::new();
                    #[cfg(not(any(windows, target_os = "linux")))]
                    if let Some(mut pipe) = child.stderr.take() {
                        let _ = pipe.read_to_string(&mut stderr);
                    }
                    bail!(
                        "llama-server exited with {status} before it was ready: {}",
                        stderr
                            .lines()
                            .rev()
                            .take(12)
                            .collect::<Vec<_>>()
                            .join(" | ")
                    );
                }
                if http::agent()
                    .get(&format!("{base}/health"))
                    .timeout(Duration::from_secs(5))
                    .call()
                    .is_ok()
                {
                    break;
                }
                if started.elapsed() > READY_TIMEOUT {
                    let _ = child.kill();
                    bail!("llama-server did not become ready within {READY_TIMEOUT:?}");
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }

        Ok(Self {
            child,
            #[cfg(windows)]
            idle_trim,
            base,
            system_prompt: SYSTEM_PROMPT.trim().to_string(),
            dictionary,
        })
    }

    /// Call only after warm-up or successful text insertion, never in correct().
    /// The exclusive borrow and synchronous call exclude a concurrent request.
    #[cfg(windows)]
    pub fn trim_idle(&mut self) {
        use std::io::Write;

        let child = &mut self.child;
        if let Some(error) = self.idle_trim.attempt(|| child.trim_working_set()) {
            // One bounded, text-free diagnostic. Even stderr failure is ignored.
            let _ = writeln!(
                std::io::stderr().lock(),
                "phonon: owned llama-server EmptyWorkingSet failed (OS error {:?}); further failures suppressed",
                error.raw_os_error()
            );
        }
    }

    /// Token count from the loaded model's own tokeniser, or an estimate when the
    /// endpoint is unavailable.
    fn count_tokens(&self, text: &str) -> u32 {
        let Ok(response) = http::agent()
            .post(&format!("{}/tokenize", self.base))
            .timeout(Duration::from_secs(10))
            .send_json(serde_json::json!({ "content": text }))
        else {
            return estimate_tokens(text);
        };
        let Ok(value) = response.into_json::<serde_json::Value>() else {
            return estimate_tokens(text);
        };
        value
            .get("tokens")
            .and_then(|tokens| tokens.as_array())
            .map(|tokens| tokens.len() as u32)
            .unwrap_or_else(|| estimate_tokens(text))
    }

    /// Ask the model once. `messages` is sent as it is.
    fn chat(&self, messages: serde_json::Value, max_tokens: u32) -> Result<String> {
        let response = http::agent()
            .post(&format!("{}/v1/chat/completions", self.base))
            .timeout(Duration::from_secs(300))
            .send_json(serde_json::json!({
                "messages": messages,
                "temperature": 0.0,
                "top_k": 1,
                // A GGUF can carry sampling defaults from the model's
                // generation_config (general.sampling.*), and llama-server
                // applies them. LFM3 ships repetition_penalty 1.1, which makes a
                // copy-editor drop repeated words: 0.098 against 0.073 WER on
                // real dictation. A formatter wants plain greedy decoding.
                "repeat_penalty": 1.0,
                "max_tokens": max_tokens,
                "stream": false,
                "cache_prompt": true,
                // Gemma 4 opens a thought channel on its own. llama.cpp turns
                // the template's thinking on by default, and the model then
                // spends the whole budget reasoning and answers nothing. A
                // formatter has nothing to reason about, so turn it off. This is
                // what the closed-thought prefill does in the macOS sidecar.
                "chat_template_kwargs": { "enable_thinking": false },
            }))?;
        let value: serde_json::Value = response.into_json().context("read correction reply")?;
        parse_reply(&value)
    }

    /// Correct one transcript.
    pub fn correct(&self, raw: &str) -> Result<String> {
        let user = dictionary_user_message(&self.dictionary, raw);
        let max_tokens = output_budget(self.count_tokens(transcript_payload(&user)));

        let separate = serde_json::json!([
            { "role": "system", "content": self.system_prompt },
            { "role": "user", "content": user },
        ]);
        let reply = match self.chat(separate, max_tokens) {
            Ok(reply) => reply,
            // Some Gemma chat templates reject a separate system turn. Fold the
            // prompt into the user turn and ask again; the text is identical.
            Err(_) => {
                let merged = serde_json::json!([{
                    "role": "user",
                    "content": format!("{}\n\n{}", self.system_prompt, user),
                }]);
                self.chat(merged, max_tokens)
                    .context("correction request failed")?
            }
        };

        let cleaned = clean_output(transcript_payload(&reply));
        let cleaned = self.dictionary.apply_exact_replacements(&cleaned).text;
        if cleaned.is_empty() {
            bail!("correction returned nothing for {raw:?}; the reply was {reply:?}");
        }
        Ok(cleaned)
    }
}

impl Corrector {
    pub fn start() -> Result<Self> {
        let dictionary = crate::dictionary::load()?;
        match backend()? {
            CorrectionBackend::Local => Ok(Self::Local(LocalCorrector::start(dictionary)?)),
            CorrectionBackend::Gmi => Ok(Self::Gmi(GmiCorrector::start(dictionary)?)),
        }
    }

    pub fn correct(&self, raw: &str) -> Result<String> {
        match self {
            Self::Local(corrector) => corrector.correct(raw),
            Self::Gmi(corrector) => corrector.correct(raw),
        }
    }

    #[cfg(windows)]
    pub fn trim_idle(&mut self) {
        if let Self::Local(corrector) = self {
            corrector.trim_idle();
        }
    }

    pub fn redacts_text(&self) -> bool {
        matches!(self, Self::Gmi(_))
    }
}

impl GmiCorrector {
    fn start(dictionary: DictionaryFile) -> Result<Self> {
        let api_key = std::env::var(GMI_API_KEY_ENV)
            .map_err(|_| anyhow!("GMI correction requires the GMI_API_KEY environment variable"))?;
        if api_key.trim().is_empty() {
            bail!("GMI_API_KEY is empty");
        }
        let model = std::env::var(GMI_MODEL_ENV).unwrap_or_else(|_| GMI_MODEL.to_string());
        if model.trim().is_empty() || model.len() > 200 || model.chars().any(char::is_whitespace) {
            bail!("{GMI_MODEL_ENV} must be a non-empty model id without whitespace");
        }
        Ok(Self {
            api_key,
            model,
            dictionary,
        })
    }

    fn request(&self, user: &str) -> Result<String> {
        let response = http::gmi_agent()
            .post(GMI_CHAT_URL)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .timeout(GMI_TIMEOUT)
            .send_json(serde_json::json!({
                "model": self.model,
                "messages": [
                    { "role": "system", "content": SYSTEM_PROMPT.trim() },
                    { "role": "user", "content": user }
                ],
                "temperature": 0.0,
                "max_tokens": GMI_MAX_TOKENS,
                // Gemini 3.8 Flash does not support reasoning disabled.
                // LOW is its smallest supported thinking level.
                "reasoning_effort": "low",
                "stream": false
            }));
        let response = match response {
            Ok(response) => response,
            Err(ureq::Error::Status(status, response)) => {
                let message = response
                    .into_json::<serde_json::Value>()
                    .ok()
                    .and_then(|value| {
                        value
                            .get("error")
                            .and_then(|error| error.get("message"))
                            .and_then(|message| message.as_str())
                            .map(str::to_owned)
                    })
                    .unwrap_or_else(|| "request was rejected".to_string());
                let message = message.chars().take(256).collect::<String>();
                bail!("GMI correction rejected with HTTP {status}: {message}")
            }
            Err(_) => bail!("GMI correction request failed"),
        };
        let mut reader = std::io::Read::take(response.into_reader(), GMI_MAX_RESPONSE_BYTES);
        let value: serde_json::Value =
            serde_json::from_reader(&mut reader).context("GMI correction returned invalid JSON")?;
        parse_cloud_reply(&value)
    }

    fn correct(&self, raw: &str) -> Result<String> {
        let mut output = String::new();
        for chunk in cloud_chunks(raw) {
            let cleaned = self.correct_chunk(&chunk)?;
            if !output.is_empty() {
                output.push(' ');
            }
            output.push_str(&cleaned);
        }
        let output = self.dictionary.apply_exact_replacements(&output).text;
        if output.trim().is_empty() {
            bail!("GMI correction returned an empty answer");
        }
        Ok(output)
    }

    /// Retry a dense chunk at half size when the provider reaches its hard
    /// output cap. This keeps normal dictations to one request per chunk while
    /// making the 128-token provider limit safe for technical speech.
    fn correct_chunk(&self, chunk: &str) -> Result<String> {
        let user = dictionary_user_message(&self.dictionary, chunk);
        match self.request(&user) {
            Ok(reply) => {
                let cleaned = clean_output(transcript_payload(&reply));
                if cleaned.is_empty() {
                    bail!("GMI correction returned an empty answer");
                }
                Ok(cleaned)
            }
            Err(error) if error.downcast_ref::<GmiTruncated>().is_some() => {
                let words = chunk.split_whitespace().collect::<Vec<_>>();
                if words.len() <= 1 {
                    return Err(error);
                }
                let middle = words.len() / 2;
                let left = self.correct_chunk(&words[..middle].join(" "))?;
                let right = self.correct_chunk(&words[middle..].join(" "))?;
                Ok(format!("{left} {right}"))
            }
            Err(error) => Err(error),
        }
    }
}

fn cloud_chunks(text: &str) -> Vec<String> {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .chunks(GMI_CHUNK_WORDS)
        .map(|chunk| chunk.join(" "))
        .collect()
}

fn parse_cloud_reply(value: &serde_json::Value) -> Result<String> {
    let finish = value
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(|reason| reason.as_str())
        .unwrap_or("unknown");
    if finish == "length" {
        return Err(anyhow::Error::new(GmiTruncated));
    }
    parse_reply(value)
}

impl Drop for LocalCorrector {
    fn drop(&mut self) {
        #[cfg(windows)]
        self.child.stop();
        #[cfg(not(windows))]
        {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_answer_out_of_a_reply() {
        let reply = serde_json::json!({
            "choices": [{ "finish_reason": "stop",
                          "message": { "content": "Hello, Fluid Voice." } }]
        });
        assert_eq!(parse_reply(&reply).unwrap(), "Hello, Fluid Voice.");
    }

    /// The first end-to-end run on Windows failed exactly this way, so the
    /// message has to name the cause rather than say "nothing came back".
    #[test]
    fn reasoning_without_an_answer_is_named() {
        let reply = serde_json::json!({
            "choices": [{ "finish_reason": "length",
                          "message": { "content": "",
                                       "reasoning_content": "Thinking Process: ..." } }]
        });
        let error = parse_reply(&reply).unwrap_err().to_string();
        assert!(error.contains("reasoned instead of answering"), "{error}");
        assert!(error.contains("length"), "{error}");
    }

    #[test]
    fn an_empty_reply_is_an_error() {
        let reply = serde_json::json!({
            "choices": [{ "finish_reason": "stop", "message": { "content": "  " } }]
        });
        assert!(parse_reply(&reply)
            .unwrap_err()
            .to_string()
            .contains("empty answer"));
        assert!(parse_reply(&serde_json::json!({})).is_err());
    }

    #[test]
    fn the_prompt_ships_inside_the_executable() {
        assert!(SYSTEM_PROMPT.contains("second-pass formatter"));
        assert!(SYSTEM_PROMPT.contains("<transcript>"));
    }

    #[test]
    fn wraps_the_transcript() {
        assert_eq!(
            user_message("  hello there  "),
            "<transcript>\nhello there\n</transcript>"
        );
    }

    #[test]
    fn reads_the_payload_back_out() {
        assert_eq!(transcript_payload("<transcript>\nhi\n</transcript>"), "hi");
        assert_eq!(transcript_payload("plain text"), "plain text");
        assert_eq!(transcript_payload("<transcript>\nunclosed"), "unclosed");
    }

    #[test]
    fn strips_leaked_channel_markers() {
        assert_eq!(clean_output("<|channel|>Hello there."), "Hello there.");
        assert_eq!(
            clean_output("<channel>thought</channel>Done."),
            "thoughtDone."
        );
        assert_eq!(clean_output("  Plain answer.  "), "Plain answer.");
        assert_eq!(clean_output("<|think|>ok<|/think|>"), "ok");
    }

    /// A less-than sign that is not a marker must survive.
    #[test]
    fn keeps_ordinary_angle_brackets() {
        assert_eq!(
            clean_output("use a < b in the loop"),
            "use a < b in the loop"
        );
        assert_eq!(clean_output("<div> stays"), "<div> stays");
    }

    #[test]
    fn budget_follows_the_macos_rule() {
        // Floor.
        assert_eq!(output_budget(0), OUTPUT_TOKEN_FLOOR);
        assert_eq!(output_budget(10), OUTPUT_TOKEN_FLOOR);
        // ceil(20 * 1.8) + 24 = 60.
        assert_eq!(output_budget(20), 60);
        // A 400-token dictation is no longer cut: ceil(400 * 1.8) + 24 = 744.
        assert_eq!(output_budget(400), 744);
        // Ceiling.
        assert_eq!(output_budget(4000), OUTPUT_TOKEN_CEILING);
    }

    #[test]
    fn the_estimate_is_never_zero() {
        assert_eq!(estimate_tokens(""), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }

    #[test]
    fn backend_defaults_to_gmi_only_when_a_key_exists() {
        assert_eq!(
            parse_backend(None, false).unwrap(),
            CorrectionBackend::Local
        );
        assert_eq!(parse_backend(None, true).unwrap(), CorrectionBackend::Gmi);
        assert_eq!(
            parse_backend(Some("local"), true).unwrap(),
            CorrectionBackend::Local
        );
        assert_eq!(
            parse_backend(Some("cloud"), false).unwrap(),
            CorrectionBackend::Gmi
        );
        assert!(parse_backend(Some("other"), true).is_err());
    }

    #[test]
    fn cloud_chunks_bound_long_transcripts_without_empty_parts() {
        let text = (1..=GMI_CHUNK_WORDS * 2 + 1)
            .map(|number| format!("w{number}"))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = cloud_chunks(&text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].split_whitespace().count(), GMI_CHUNK_WORDS);
        assert_eq!(chunks[1].split_whitespace().count(), GMI_CHUNK_WORDS);
        assert_eq!(chunks[2].split_whitespace().count(), 1);
    }

    #[test]
    fn cloud_length_replies_are_rejected_even_when_content_exists() {
        let reply = serde_json::json!({
            "choices": [{ "finish_reason": "length",
                          "message": { "content": "partial answer" } }]
        });
        let error = parse_cloud_reply(&reply).unwrap_err().to_string();
        assert!(error.contains("truncated"), "{error}");
        assert!(!error.contains("partial answer"), "{error}");
    }

    #[test]
    fn correction_threads_leave_one_core_free() {
        assert_eq!(thread_count(1), 1);
        assert_eq!(thread_count(2), 1);
        assert_eq!(thread_count(16), 15);
    }
}
