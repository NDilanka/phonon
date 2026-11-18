# Phonon

Open-source voice typing. Fast. Local. Yours.
macOS is the shipped platform, and that tagline describes it. Windows is in
beta: its cloud correction path keeps audio local but sends transcript text to
a hosted model, so Windows cloud correction is not fully local (see
"Windows beta").

[Website](https://phonon.sh) · [Releases](https://github.com/Infatoshi/phonon/releases)
(upstream links. The Windows beta source is the fork at https://github.com/NDilanka/phonon, based on Infatoshi/phonon.)

## Install

Apple silicon, macOS 14 or later.

```bash
brew install --cask infatoshi/tap/phonon
phonon
```

The cask installs the Developer ID-signed and notarized app. On first launch
Phonon downloads Parakeet TDT 0.6B v2 for speech recognition and the Phonon
refiner 1.7B for correction, both pinned to exact revisions. That is about
4.9 GB (2.47 GB for Parakeet, 2.40 GB for the refiner) plus the Python packages
`uv` installs, and the one slow start; everything after it is local and offline, apart from a
once-a-day check of GitHub for a newer release, which carries nothing about you and can be turned off
in Settings. Tagged releases
also include the DMG for direct installation. To build from source instead, use
`brew install --formula infatoshi/tap/phonon`.

## The recipe

This is the macOS recipe. The Windows beta uses different models and an
optional cloud correction backend; see "Windows beta".

- **Hear:** Parakeet TDT 0.6B v2 transcribes on your Mac, on Phonon's own Metal kernels.
- **Rewrite:** the Phonon refiner 1.7B, fine-tuned from LiquidAI's LFM3-1.7B, turns the transcript into
  what you meant to type: misheard terms fixed, filler and false starts gone, only the final version kept
  when you correct yourself.
- **Data:** 76,960 transcripts of public audio and synthetic dictations, each rewritten by DeepSeek
  following a written spec, in four writing modes (casual, formal, prompt, minimal). No personal dictation
  went into it. The pipeline is in `ml/`.
- **Mode:** picked from the app you dictate into: casual in Discord, formal in Slack and email, prompt in
  AI assistants and terminals, minimal everywhere else.
- **Speed:** both models work while you talk, so at key-up only the last seconds are left. The refiner runs
  on a native Metal engine (a decode megakernel with a speculative drafter). Key-up to pasted text is about
  65 ms on an M4 Max.

## Windows beta

Beta. Continuous integration builds the Windows crate on pushes to `main` and
`windows-beta`, and runs one selftest pass through the default local backend.
This fork's
Windows work is source-only for now: there is no signed or prerelease Windows
binary, and this fork's source is at https://github.com/NDilanka/phonon.
Upstream publishes
its own releases at
[github.com/Infatoshi/phonon/releases](https://github.com/Infatoshi/phonon/releases);
those upstream artifacts do not contain this fork's Windows changes.

The instructions below cover the cloud correction path on Windows 10 or 11
(x64). No GPU, CUDA, Python, or WSL is needed for it.

### What the cloud path keeps and sends

- Audio is captured and recognized on your PC. Recordings are kept as local
  WAV files; there is no automatic retention cleanup.
- Transcript chunks and your dictionary guidance (including custom entries)
  are sent over HTTPS to GMI, a hosted inference provider, for correction. You
  need your own GMI API key, credit, and access to the model used.
- This path is not fully local, fully private, or zero-retention, and it is a
  beta, not turnkey. Error paths can include provider response text in logs
  and tray messages. Clipboard-based insertion can expose corrected text to
  clipboard history and clipboard managers.
- A cloud failure never silently starts a local model and never inserts the
  raw speech-recognition result.
- Correction requests currently use `google/gemini-3.8-flash` at temperature 0
  with low reasoning effort and a 512-token output cap per request (see
  `crates/phonon-win/src/polish.rs`). Text is sent in chunks of up to 64
  whitespace-separated words, which loses paragraph breaks, and a truncated
  chunk may be split and retried. These are implementation settings, not a
  promise of perfect punctuation, fixed cost, or fixed latency.

### Prerequisites

- PowerShell 7.
- Stable Rust and Cargo for `x86_64-pc-windows-msvc` (for example via rustup).
- Visual Studio Build Tools with the "Desktop development with C++" workload,
  including the Windows SDK.
- Microsoft Visual C++ 2015-2022 Redistributable (x64), unless already
  installed. The downloaded sherpa-onnx and ONNX Runtime binaries link the
  shared MSVC runtime and need it even though Phonon's own executable links
  the CRT statically.
- Your own GMI API key with access to `google/gemini-3.8-flash`.
- Network access to crates.io, GitHub, and Hugging Face for the build and the
  first speech download, and to GMI (`api.gmi-serving.com`) for every
  non-empty correction.
- Several gigabytes of free disk. The first speech download transfers about
  648 MiB, downloaded archives are kept next to the installed copies, and the
  Rust build output comes on top, so total disk use is well above the transfer
  size.

### Build

From the repository root, in PowerShell 7:

```powershell
cargo build --release --locked -p phonon-win
if ($LASTEXITCODE -ne 0) { throw 'Build failed. Stop here.' }
```

Stop here if the build fails. The launch scripts never build or update the
executable; rebuild it yourself after any source change. The executable you
build is not code signed.

`cargo test --workspace` has two known failures on Windows, in crates this
fork does not change: `phonon-mine` (`fixture_tree_pins_exact_extract_output`,
a CRLF and entry-order fixture) and `phonon-profile`
(`e2e_report_keeps_dictation_time_out_of_release_latency`, a temporary file
name containing `::`). The other test targets pass.

### Fetch, self-test, and run

Use a private PowerShell 7 window, with no transcription or recording of the
terminal. The key is prompted securely, exposed as plain text only inside this
shell's process environment, and removed again when the tray exits; it is not
written into the command line, shell history, a file, or the registry:

```powershell
$secureKey = Read-Host 'GMI API key' -AsSecureString
if ($secureKey.Length -eq 0) { throw 'No key supplied.' }
$env:GMI_API_KEY = [System.Net.NetworkCredential]::new('', $secureKey).Password
$env:PHONON_WIN_CORRECTION_BACKEND = 'gmi'
$env:PHONON_WIN_GMI_MODEL = 'google/gemini-3.8-flash'

try {
    pwsh -NoProfile -File .\launch-phonon.ps1 fetch
    if ($LASTEXITCODE -ne 0) { throw 'Speech asset download failed.' }

    pwsh -NoProfile -File .\launch-phonon.ps1 selftest
    if ($LASTEXITCODE -ne 0) { throw 'Startup test failed. Do not dictate yet.' }

    pwsh -NoProfile -File .\launch-phonon.ps1
    if ($LASTEXITCODE -ne 0) { throw 'Launcher failed.' }
}
finally {
    Remove-Item Env:GMI_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:PHONON_WIN_CORRECTION_BACKEND -ErrorAction SilentlyContinue
    Remove-Item Env:PHONON_WIN_GMI_MODEL -ErrorAction SilentlyContinue
    $secureKey.Dispose()
}
```

Setting the backend and model explicitly matters: a present `GMI_API_KEY`
would select cloud anyway, but an explicit backend prevents accidentally
loading a local model. The launcher also clears proxy environment variables
for its child process, so proxy-required networks are untested.

The tray command waits until you quit Phonon from its tray menu. Leave the
shell open until then so the cleanup block runs, and re-run this prompt in
each new session.

`fetch` downloads only the pinned sherpa-onnx runtime and the Parakeet speech
model (about 648 MiB); in GMI mode it does not download the Gemma correction
model or llama.cpp. `fetch` alone does not test your GMI access. `selftest`
recognizes a bundled public sample and sends that sample's transcript plus
dictionary guidance to GMI, so it doubles as the check that your key and model
access work. Starting the tray repeats the same startup check.

The launcher always uses `<checkout>\research\laptop\data` as the data folder,
creating it on first use, and overrides any `PHONON_WIN_HOME` you may have
set. Models, the dictionary, and recordings all live there. Running
`target\release\phonon-win.exe` directly instead defaults to
`%LOCALAPPDATA%\Phonon`, which would create a second, separate cache and
dictionary; stick to the launcher. Delete the data folder to start over.

### Dictate

1. Allow Windows microphone access for desktop apps.
2. Find Phonon's tray icon (it may sit behind the **^** hidden-icons arrow at
   the right of the taskbar) and wait for **ready**.
3. Focus a normal writable text field.
4. Hold **Right Ctrl**, speak, then release. Double-tap Right Ctrl to latch
   recording; press it again to stop.
5. Start with a harmless short sentence and verify the inserted text. Quit
   through the tray menu when done.

Right-click the tray icon for the menu. Phonon swallows Right Ctrl while it
runs, so Right Ctrl stops acting as a modifier; Left Ctrl is untouched.
`PHONON_WIN_HOTKEY` takes `rightctrl`, `leftctrl`, `rightalt`, `rightshift`,
`capslock`, or `f13`.

### Local correction (CPU route; larger download)

Fully offline local correction exists but is constrained in this fork: the
default local branch of the launcher expects a laptop-specific CUDA runtime
folder that a fresh checkout does not contain, so on a new machine the
practical local route is CPU:

```powershell
$env:PHONON_WIN_CORRECTION_BACKEND = 'local'
pwsh -NoProfile -File .\launch-phonon.ps1 -Cpu
```

Set the backend to `local` explicitly: `-Cpu` alone does not choose the
backend, and a configured GMI key keeps cloud correction active. Local mode
additionally downloads the llama.cpp CPU build and the larger Gemma correction
weights. The tray tooltip carries download progress.

In local mode, after startup warm-up and after each successful correction and
text insertion, Phonon trims only its own local correction process. The model
stays loaded: this releases resident RAM, not model allocations or VRAM, and
the next correction may page data back in. In cloud mode there is no local
correction process at all. Trimming is best effort, after input is queued,
never on the key-release-to-insertion path.
Set `$env:PHONON_WIN_TRIM_IDLE = '0'` before launching to disable it; unset or `1`
enables it. Other values are rejected before starting the correction server.
Both local launchers (`launch-phonon.ps1` and `launch-phonon-cpu.ps1`) preserve
this setting. Windows `bench` and `selftest` also trim after each corrected pass.
Benchmark timings exclude trimming; each pass is printed before trimming starts.

The executable also has direct subcommands (the launchers forward arguments):

```
phonon-win.exe info      # what first run downloads, and what is installed
phonon-win.exe fetch     # download and verify, then stop
phonon-win.exe selftest  # one pass through ASR and the active correction backend
```

`bench` is intentionally local-only and refuses the GMI backend.

### Custom dictionary

Windows stores `dictionary.json` in the active data folder. Under the launchers
that is `<checkout>\research\laptop\data\dictionary.json`, because the
launchers force `PHONON_WIN_HOME` there; a direct `phonon-win.exe` run defaults
to `%LOCALAPPDATA%\Phonon\dictionary.json`, or `PHONON_WIN_HOME` when that
override is set. It uses the same JSON format as macOS. The tray, `bench`, and `selftest` load it once at engine startup;
restart the tray after editing. A malformed file stops startup with its path
in the error instead of discarding your edits.

If missing, the file is created with Phonon, Droid, Factory, Factory Desktop,
WSL, GB, cargo, PostgreSQL, and GitHub. Existing files are never overwritten.
Keep one entry per term; `phrase` is its written spelling. Optional
`spoken_forms` lists explicit misheard forms, and optional `replacement` maps
the phrase and its spoken forms to different output text:

```json
{
  "schema_version": 1,
  "updated_at_unix_ms": 0,
  "entries": [
    {
      "phrase": "Phonon",
      "spoken_forms": ["foe non"],
      "source": "user",
      "starred": false,
      "usage_count": 0
    },
    {
      "phrase": "work in progress",
      "replacement": "WIP",
      "spoken_forms": [],
      "source": "user",
      "starred": false,
      "usage_count": 0
    }
  ]
}
```

The dictionary supplies relevant terms and conservative pre-corrections to the
correction model; explicit replacements are also enforced on its cleaned output.
ASR hotwords are not enabled by this Windows dictionary support. Phonetic guessing
uses the shared `assets/english_words.txt` ordinary-word guard, embedded in the
Windows executable at build time. No separate list or source checkout is needed
at runtime; a missing list at build time is a compilation error.

## Pipeline

```
mic → Parakeet speech recognition → dictionary lookup → active correction backend → typed into the app
```

The active backend loads at launch. The startup screen reaches 100% only
after Parakeet transcribes a bundled clip through both the batch and the
streaming path, that transcript goes through the correction model, and a
representative technical correction succeeds. There is no speech-only mode.

On macOS, the correction model is the Phonon refiner 1.7B, fine-tuned from
LiquidAI's LFM3-1.7B on transcripts of public audio and synthetic transcripts;
no personal dictation went into it. It writes what you meant to type: misheard terms
fixed, filler and false starts gone, and only the final version kept when you
correct yourself. Its writing mode follows the app you dictate into: casual in
personal chat (Discord, Messages, WhatsApp), formal in email and work chat
(Slack, Teams), prompt in AI assistants, terminals and editors, minimal
everywhere else. It runs on
Phonon's native Metal engine (a decode megakernel with a speculative drafter,
prefill through a patched llama.cpp), and falls back to an MLX copy of the same
model when the engine cannot run. Parakeet also runs on Phonon's own Metal
kernels, with MLX as the fallback. On Macs with 16 GB or more, both stages work
while you talk: Parakeet re-transcribes the recording every 0.4 s and the
refiner corrects the transcript so far, so at key-up only the last seconds are
transcribed again. On an M4 Max that took the median from key release to
corrected text from about 215 ms to about 30 ms (40 real dictations, measured in
the engine; pasting adds a few ms), with the same text as transcribing the whole
recording at once. While the key is held, a card above the capsule shows the
corrected text so far and the words just heard after it. It changes only while
you speak, and words more than a dozen back stay put until key-up (Settings
turns it off). Dictations of any length stream: past about 200 words the older
sentences are corrected once and kept, so key-up stays short.
`sidecar/polish_server.py` drives macOS correction. On Windows, the active
backend is local Gemma or GMI Cloud, selected by the launcher and environment.

## Build from source

```bash
cargo install --path crates/phonon-cli --force --root ~/.local
phonon bar --rebuild   # once: builds and signs the app from this checkout
```

## Commands

```bash
phonon                       # open the app
phonon doctor                # check the runtime, models and data files
phonon bench                 # time speech recognition and correction
phonon profile model         # warm correction timings by input length
phonon profile e2e           # key-press-to-insertion timings from recent dictations
phonon dictionary --help     # terms, replacements, Wispr Flow import, evaluation
phonon dictionary import-txt # add one term per line from a text file
phonon corpus --help         # stored recordings and their transcripts
phonon stats                 # words, sessions, speaking time, dictionary fixes
```

## Surviving an uninstall

Uninstallers match on the app name, the developer name, or the bundle
identifier, then delete every hit under `~/Library/Application Support`,
`Preferences`, `Caches`, `Logs`, and `Saved Application State`. Nothing kept
there can survive one.

So Phonon mirrors the small irreplaceable files to `~/.phonon/backup`, which no
such pattern reaches: the dictionary, settings, history, vocabulary and word
replacements, about 128 KB. The mirror is rewritten whenever any of them change,
and a wiped store can never overwrite a populated backup.

If Phonon ever starts with an empty data folder while that backup exists, it
asks whether to restore it, and offers to delete it instead. It never restores
silently. Settings › Backup shows what is held, reveals it in Finder, deletes
it, and exports everything including the corpus.

Model weights are unaffected. They live in `~/.cache/huggingface`, are named
after the models rather than the app, and no uninstaller touches them.

## Local data

Nothing is retained until you say so. On first launch the app asks once whether
to keep local history and whether to read the screen for context; both start off,
and declining leaves dictation fully working. Settings carries the retention
window (keep until deleted, or 7 / 30 / 90 days) and a clear-everything button.

With local history on, every recording is kept with its metadata:

```text
~/Library/Application Support/Phonon/
├── dictionary.json
├── settings.json
└── Corpus/
    └── <source>_<timestamp>_<process>/
        ├── audio.wav
        └── metadata.json
```

`metadata.json` keeps the raw transcript, the corrected text, what you meant
(when you save it in History), dictionary changes, source, duration, and
correction timings.

Before speech recognition, an adaptive audio gate requires sustained
speech-like energy above the clip's measured noise floor. A clip without speech
stays in the corpus with `speech_detected: false`, skips both models, and
produces no text.

Working with the dictionary and the stored recordings:

```bash
phonon dictionary import-wispr
phonon dictionary import-txt
phonon dictionary test 'run v llm on black well'
phonon dictionary evaluate
phonon corpus list --search Blackwell
phonon corpus show <id>
phonon corpus set-intended <id> 'what you meant to say'
phonon dictionary learn <id> --from 'black well' --to Blackwell
```

With screen context on, Phonon reads the windows visible on each display when
a recording starts, plus any window brought to the front while you speak, with
macOS Vision OCR on the Mac. Screenshots are never saved. The OCR text only
confirms dictionary terms that already match the spoken transcript; those
terms, not the screen text, go into the correction prompt. Apps and sites on
the private list (Settings › Private apps and sites) are never read, and
password managers are always private.

### Floating bar

- One capsule at the bottom of the screen; it widens with your voice while
  listening and breathes while the text is corrected.
- Hold Globe (fn) to talk and release to type into the frontmost app.
- Double-tap Globe to keep recording; tap once to stop and type.

Globe (fn) is the default shortcut. Settings can switch the hold key to Right
Option (⌥), and either hold key can be paired with a Control-Space toggle.
macOS also acts on the Globe key, so set System Settings › Keyboard ›
"Press 🌐 key to" to "Do Nothing".

Recording cues are off by default. Turning them on plays a short rising sweep
when a recording opens and a falling one when it closes. Both are synthesized by
`scripts/make-cue-sounds.py`; no audio is sampled or downloaded.

Phonon needs Microphone, Accessibility and Input Monitoring access, plus Screen
Recording if screen context is on.

### Native app

`phonon` opens a macOS Dock app (Settings › System › Show in Dock off keeps it in
the menu bar only). Its Home, History, Dictionary, and
Settings pages read and write the same JSON files and recordings as the engine
and the CLI. Settings › Model status shows whether both models are loaded. The
`ϕ` in the menu bar stays available as a shortcut while the app runs.

Phonon uses the input selected in System Settings, except a Bluetooth headset
microphone, which would drop the headset to call quality. A priority list on
Home overrides that. Availability is checked before each recording, so a
preferred external microphone takes over as soon as it is connected. The menu
bar menu shows the microphone in use.

## Layout

```
crates/phonon-audio/   recording and audio files
crates/phonon-asr/     Parakeet sidecar lifecycle and protocol
crates/phonon-llm/     correction sidecar lifecycle and benchmark client
crates/phonon-profile/ Metal dispatch, model phase, and key-to-insertion profilers
crates/phonon-core/    pipeline coordination and engine events
crates/phonon-cli/     the phonon command: app launcher, doctor, bench, data tools
crates/phonon-hotkey/  hold, tap, and double-tap latch, shared by macOS and Windows
crates/phonon-win/     Windows tray app, keyboard hook, WASAPI capture, insertion
crates/phonon-mine/    source-identifier extraction for vocabulary mining
bar/                   SwiftPM macOS app: Home, History, Dictionary, Settings, capsule
sidecar/               speech recognition and correction servers (Python, run by uv)
prompts/               correction prompt
assets/                startup clip, English word list, recording cues
ml/                    refiner training and evaluation pipeline (runs on a GPU box)
tools/                 vocabulary-mining research tools
tools/dev/             developer benchmarks that call cloud APIs; not part of the app
website/               phonon.sh
```

## License

Phonon source is GPL-3.0. Model weights are downloaded, not stored in this
repository, and keep their own licenses: Parakeet is CC-BY-4.0, and the Phonon
refiner, as a derivative of LFM3, is under the LFM Open License v1.0, which
allows commercial use only for organizations under $10M in annual revenue. See
[THIRD_PARTY.md](THIRD_PARTY.md).
