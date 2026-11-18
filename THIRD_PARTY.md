# Third-party models and runtimes

Phonon does not commit model weights to this repository. The macOS public release
downloads Parakeet and Phonon's public refiner from Hugging Face, each pinned
by repository and revision.

## Windows source-only beta

This fork publishes source, not executable releases, runtime DLLs, model weights,
recordings, or local benchmark data. First-run downloads are pinned by URL,
revision and SHA-256 in `crates/phonon-win/src/manifest.rs`. In GMI mode only
the speech runtime and Parakeet model are downloaded; correction uses the
user's own GMI account. Audio stays local, but transcript chunks and dictionary
guidance are sent to GMI. Hosted-model access remains subject to the provider's
terms and is not a redistribution of Gemini weights.

- **sherpa-onnx 1.13.6:** [source and release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.6),
  [Apache-2.0 license](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/LICENSE).
  Windows downloads the x64 shared-MD, no-TTS archive named in the manifest.
- **ONNX Runtime:** the sherpa archive supplies its native inference DLLs;
  the audited pinned Windows archive reports version 1.27.1.
  [Source](https://github.com/microsoft/onnxruntime/tree/v1.27.1) and
  [MIT license](https://github.com/microsoft/onnxruntime/blob/v1.27.1/LICENSE).
  Preserve the archive's applicable dependency licenses and notices if distributing
  native runtime files separately; this source-only fork does not bundle them.
- **Parakeet TDT 0.6B v2, int8 ONNX:** NVIDIA's
  [original model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2),
  exported/quantized by csukuangfj at
  [`1ab9323565ddb038682214b292f588070a538ce2`](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tree/1ab9323565ddb038682214b292f588070a538ce2).
  The [pinned model card](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/blob/1ab9323565ddb038682214b292f588070a538ce2/README.md)
  declares CC-BY-4.0; model weights keep their own license and attribution,
  not the source code's GPL license.
- **Optional local correction:** Windows downloads the llama.cpp CPU runtime
  [b10726](https://github.com/ggml-org/llama.cpp/releases/tag/b10726) (MIT) and
  Google's Gemma 4 E2B QAT q4_0 GGUF at
  [`675cff42a74c774d6cb76f76d8eacb49b48c9b93`](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/tree/675cff42a74c774d6cb76f76d8eacb49b48c9b93),
  not the macOS Phonon refiner below. The exact
  [pinned model card](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/675cff42a74c774d6cb76f76d8eacb49b48c9b93/README.md)
  declares Apache-2.0 and links Google's
  [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license).
  Consult those exact model terms and notices rather than inferring terms
  for all Gemma models or applying the unrelated LFM refiner license.

## uv

- Source: <https://github.com/astral-sh/uv>
- License: Apache-2.0 OR MIT

The signed DMG bundles the arm64 `uv` executable so a clean Mac can create the
local Python environment used by the sidecars. Model weights and Python
packages are still downloaded into the user's normal caches on first launch.

## Parakeet TDT 0.6B v2

- Source: <https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2>
- Upstream model: NVIDIA Parakeet TDT 0.6B v2,
  <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2>
- License: CC-BY-4.0. Converted to MLX by mlx-community; Phonon downloads it
  unchanged from that repository.
- Primary weight: `model.safetensors`
- Size: 2,471,559,904 bytes
- SHA-256: `b958c37a6baa6874a279108755c8f2818e27bf647d72d54800a234a421341dfe`

## Phonon refiner 1.7B

- Source: <https://huggingface.co/Infatoshi/phonon-refiner-1.7b>
- Base model: LiquidAI LFM3-1.7B, <https://huggingface.co/LiquidAI/LFM3-1.7B>
- License: LFM Open License v1.0 (the repository carries the full text). It
  permits redistribution of fine-tuned weights. Commercial use is licensed
  only to organizations under 10 million US dollars in annual revenue; a
  larger organization's commercial use of these weights is not licensed by it.
- Training data: public YouTube technical audio and programming livestreams
  transcribed by Parakeet, and synthetic dictation-style transcripts, labeled by
  DeepSeek-V4.1-Flash-1017 (MIT), run locally for the 0.2.3 rows and through
  DeepSeek's hosted API for the writing-mode rows, plus synthetic term-repair
  rows. No personal dictations. The audio is not redistributed.
- Files: the engine's GGUF target and DFlash2 drafter, and the MLX weights
  used when the engine cannot run. The model card lists sizes and checksums.

## llama.cpp and ggml

- Source: <https://github.com/ggml-org/llama.cpp>, commit
  `39febd2adb947b70da35e95f9351b3bf58ad170a`
- License: MIT (`third_party/llama.cpp/LICENSE`)
- Changes: `third_party/llama.cpp/phonon-engine.patch`

The native refiner engine (`engine/`, bundled as `Contents/Helpers/pe`)
statically links llama.cpp's tokenizer, prompt prefill and drafter code. The
app carries the license at `Contents/Resources/licenses/llama.cpp/`, together
with the license of nlohmann/json (MIT, `LICENSE-jsonhpp`), which llama.cpp
vendors.

## mlx-lm LFM3 model definition

- Source: <https://github.com/ml-explore/mlx-lm/pull/2016>, commit `9027ac5b0`
- License: MIT (Copyright Apple Inc.)
- File: `sidecar/models/lfm3.py`, unchanged apart from a provenance comment

The pinned mlx-lm release cannot load LFM3. The sidecar registers this file only
when the installed mlx-lm has no `lfm3` module, so it drops out on its own once a
pinned release includes it.

## Training-time tools (not shipped)

- SpecForge, <https://github.com/sgl-project/SpecForge>, MIT: the drafter's
  training objective and optimizer.
- DFlash, <https://github.com/z-lab/dflash>, MIT: the block drafter design the
  DFlash2 drafter extends.
- DeepSeek-V4.1-Flash-1017,
  <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017>, MIT: the
  labeling model, run locally. DeepSeek's hosted API was not used.
- ShareGPT_V3_unfiltered_cleaned_split, from
  <https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered>
  (listed as Apache-2.0): input text for the drafter only, labeled with the
  refiner's own outputs.

## Fluid-1 (developer comparison only)

- Source: <https://huggingface.co/ALTICDEV/FLUID-1>
- License: AGPL-3.0

FluidVoice itself is GPL-3.0, but its release-only `fluid-intelligence-mlx`
provider is not present in the public FluidVoice source. Phonon does not
redistribute that helper or download Fluid-1; `phonon bench --baseline` uses a
local FluidVoice install when one exists.
