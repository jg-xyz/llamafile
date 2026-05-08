# llamafile

<img src="docs/images/llamafile-640x640.png" width="320" height="320"
     alt="[line drawing of llama animal head in front of slightly open manilla folder filled with files]">

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/mozilla-ai/llamafile/blob/main/LICENSE)
[![ci status](https://github.com/mozilla-ai/llamafile/actions/workflows/ci.yml/badge.svg)](https://github.com/mozilla-ai/llamafile/actions/workflows/ci.yml)
[![Based on llama.cpp](https://img.shields.io/badge/llama.cpp-7f5ee54-orange.svg)](https://github.com/ggml-org/llama.cpp/commit/7f5ee54)
[![Based on whisper.cpp](https://img.shields.io/badge/whisper.cpp-2eeeba5-green.svg)](https://github.com/ggml-org/whisper.cpp/commit/2eeeba5)
[![Discord](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr)
[![Mozilla Builders](https://img.shields.io/badge/Builders-6E6E6E?logo=mozilla&logoColor=white&labelColor=4A4A4A)](https://builders.mozilla.org/)

**llamafile lets you distribute and run LLMs with a single file.**

llamafile is a [Mozilla Builders](https://builders.mozilla.org/) project (see its [announcement blog post](https://hacks.mozilla.org/2023/11/introducing-llamafile/)), now revamped by [Mozilla.ai](https://www.mozilla.ai/open-tools/llamafile). 

Our goal is to make open LLMs much more
accessible to both developers and end users. We're doing that by
combining [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one
framework that collapses all the complexity of LLMs down to
a single-file executable (called a "llamafile") that runs
locally on most operating systems and CPU archiectures, with no installation.

llamafile also includes **[whisperfile](docs/whisperfile/index.md)**, a single-file speech-to-text tool built on [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and the same Cosmopolitan packaging. It supports transcription and translation of audio files across all the same platforms, with no installation required.


## v0.10.0

**llamafile versions starting from 0.10.0 use a new build system**, aimed at keeping our code more easily 
aligned with the latest versions of llama.cpp. This means they support more recent models and functionalities,
but at the same time they might be missing some of
the features you were accustomed to (check out [this doc](README_0.10.0.md) for a high-level description of what has been done). If you liked
the "classic experience" more, you will always be able to access the previous versions from our
[releases](https://github.com/mozilla-ai/llamafile/releases) page. Our pre-built llamafiles always
show which version of the server they have been bundled with ([0.9.* example](https://huggingface.co/mozilla-ai/llava-v1.5-7b-llamafile), [0.10.* example](https://huggingface.co/mozilla-ai/llamafile_0.10.0)), so you will always know
which version of the software you are downloading.


> **We want to hear from you!**
Whether you are a new user or a long-time fan, please share what you find most valuable about llamafile and what would make it more useful for you.
[Read more via the blog](https://blog.mozilla.ai/llamafile-returns/) and add your voice to the discussion [here](https://github.com/mozilla-ai/llamafile/discussions/809).


## Quick Start

Download and run your first llamafile in minutes:

```sh
# Download an example model (Qwen3.5 0.8B)
curl -LO https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/Qwen3.5-0.8B-Q8_0.llamafile

# Make it executable (macOS/Linux/BSD)
chmod +x Qwen3.5-0.8B-Q8_0.llamafile

# Run it
./Qwen3.5-0.8B-Q8_0.llamafile
```

We chose this model because that's the smallest one we have
built a llamafile for, so most likely to work out-of-the-box for you.
If you have powerful hardware and/or GPUs, [feel free to choose](https://mozilla-ai.github.io/llamafile/example_llamafiles/)
larger and more expressive models which should provide more accurate
responses.

**Windows users:** Rename the file to add `.exe` extension before running.

## Building a llamafile

The real power of llamafile is that you can **bundle any GGUF model into a
single self-contained executable** that runs on macOS, Linux, Windows, and BSD
with no installation, no runtime dependencies, and no configuration. You hand
someone one file; they run it.

A llamafile is an [APE](https://justine.lol/ape.html) (Actually Portable
Executable) that uses ZIP as an embedded container. The executable, model
weights, and default arguments all live in the same file. Recipients can
inspect its contents with `unzip -vl yourmodel.llamafile`.

### What you need

1. **The llamafile binary** — build from source (`mise run build`) or download
   a prebuilt release binary from the
   [releases page](https://github.com/mozilla-ai/llamafile/releases).

2. **Model weights in GGUF format** — download from
   [Hugging Face](https://huggingface.co/models?library=gguf), or convert from
   HuggingFace safetensors (see [Converting BitNet Models](#converting-bitnet-models-to-gguf)
   for 1-bit models).

3. **A `.args` file** — one argument per line, setting the default flags baked
   into the executable.

4. **zipalign** — built alongside llamafile at `o//third_party/zipalign/zipalign`.
   Build it standalone with `make o//third_party/zipalign`.

> **Note:** This is *not* the Android zipalign tool — it's a
> [different project](https://github.com/jart/zipalign) included as a submodule.

### The .args file

The `.args` file contains one flag or value per line. A few conventions:

- Model weights bundled inside the executable must be referenced as
  `/zip/<filename>` — the `/zip/` prefix tells llamafile to load from the
  embedded ZIP.
- The `...` token at the end passes through any additional CLI arguments the
  user supplies at runtime, so flags like `--server` or `-c 8192` still work.
- Arguments are evaluated before any CLI arguments, so CLI flags can always
  override the baked-in defaults.

### Example: interactive TUI chatbot

```bash
# 1. Download a GGUF model
curl -LO https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

# 2. Write a .args file
cat > .args <<'EOF'
-m
/zip/Qwen3-0.6B-Q8_0.gguf
--temp
0.6
--top-k
20
--top-p
0.95
-c
32768
--no-mmap
...
EOF

# 3. Copy the binary and embed the weights + args
cp o//llamafile/llamafile Qwen3-0.6B.llamafile
o//third_party/zipalign/zipalign -j0 Qwen3-0.6B.llamafile Qwen3-0.6B-Q8_0.gguf .args

# 4. Run it — works on any platform, no install needed
chmod +x Qwen3-0.6B.llamafile
./Qwen3-0.6B.llamafile           # combined TUI + server (default)
./Qwen3-0.6B.llamafile --chat    # TUI only
./Qwen3-0.6B.llamafile --server  # HTTP API only
./Qwen3-0.6B.llamafile --cli -p "Write a haiku"
```

### Example: server-mode with a multimodal model

Bake `--server` into `.args` so the file starts as an API server by default:

```bash
cat > .args <<'EOF'
-m
/zip/llava-v1.6-mistral-7b.Q8_0.gguf
--mmproj
/zip/mmproj-model-f16.gguf
--server
--host
0.0.0.0
-ngl
9999
--no-mmap
...
EOF

cp o//llamafile/llamafile llava.llamafile
o//third_party/zipalign/zipalign -j0 \
  llava.llamafile \
  llava-v1.6-mistral-7b.Q8_0.gguf \
  mmproj-model-f16.gguf \
  .args

./llava.llamafile             # starts HTTP server on :8080 immediately
./llava.llamafile --port 9090 # user can still override baked-in flags
```

### Example: bundling a 1-bit Bonsai model

1-bit models get Turbo1Bit KV compression auto-injected at runtime, but you can
also bake it into `.args` for maximum portability:

```bash
mise run model:download -- prism-ml/Bonsai-8B-gguf models/Bonsai-8B

cat > .args <<'EOF'
-m
/zip/Bonsai-8B.gguf
-fa
on
--ctk
q4_0
--ctv
q4_0
-c
65536
--no-mmap
...
EOF

cp o//llamafile/llamafile Bonsai-8B.llamafile
o//third_party/zipalign/zipalign -j0 \
  Bonsai-8B.llamafile \
  models/Bonsai-8B/Bonsai-8B.gguf \
  .args

./Bonsai-8B.llamafile                          # 65K context, 4-bit KV cache
./Bonsai-8B.llamafile --cli -p "Explain ternary neural networks"
```

Or use the mise task shorthand (handles `.args` generation automatically):

```bash
mise run model:bundle -- models/Bonsai-8B/Bonsai-8B.gguf Bonsai-8B.llamafile --turbo1bit
```

### Distributing your llamafile

The resulting file is fully self-contained and runs on any supported platform
without modification. Good places to host llamafiles:

- **Hugging Face** — include the llamafile git revision or release version in
  your commit message so users can verify provenance. The Apache 2.0 license
  requires noting any source changes; you can embed a notice inside the ZIP with
  `zipalign`.
- **Any file host** — the file is a valid ZIP and a valid executable simultaneously.

To inspect what's inside any llamafile (including ones you've downloaded):

```bash
unzip -vl yourmodel.llamafile
```

---

## 1-bit LLMs: Bonsai + BitNet

This fork's headline feature is **native 1-bit LLM support** — the ability to run
[Bonsai](https://github.com/PrismML-Eng/llama.cpp/tree/prism) and
[BitNet](https://github.com/microsoft/BitNet) models efficiently on any hardware,
packaged as a single portable file.

### Why 1-bit?

1-bit and 1.58-bit LLMs quantize weights to `{-1, 0, +1}` (ternary). At the same
parameter count as a standard FP16 model, this delivers:

- **~8–16x smaller model weights** — a 8B model fits in ~2 GB RAM instead of 16 GB
- **~3x faster CPU inference** — integer LUT lookups replace FP multiply-accumulate
- **Bounded KV cache memory** — with Turbo1Bit compression, long contexts (65K+
  tokens) stay within normal laptop RAM budgets

### Bonsai (recommended starting point)

Bonsai models from [PrismML](https://github.com/PrismML-Eng/llama.cpp/tree/prism)
ship as pre-converted GGUFs — no Python, no conversion step, just download and run:

```bash
# Download Bonsai-8B GGUF (~2 GB)
mise run model:download -- prism-ml/Bonsai-8B-gguf models/Bonsai-8B

# Run — Turbo1Bit KV compression auto-enabled for 1-bit filenames
llamafile -m models/Bonsai-8B/Bonsai-8B.gguf

# Long context: 65K tokens with 4-bit KV cache (fits in 16 GB RAM)
llamafile -m models/Bonsai-8B/Bonsai-8B.gguf -c 65536

# CLI mode
llamafile -m models/Bonsai-8B/Bonsai-8B.gguf --cli -p "Explain 1-bit neural networks"
```

Bonsai-1.7B is also available (`prism-ml/Bonsai-1.7B-gguf`) for even lower RAM requirements.

### BitNet (Microsoft + community models)

BitNet models from HuggingFace require a one-time GGUF conversion:

```bash
# Full pipeline: download, convert, quantize
mise run model:convert -- models/bitnet-2B --hf-repo microsoft/BitNet-b1.58-2B-4T

# Run with llamafile
llamafile -m models/bitnet-2B/ggml-model-i2_s.gguf

# Bundle as a standalone portable executable
mise run model:bundle -- models/bitnet-2B/ggml-model-i2_s.gguf BitNet-2B.llamafile --turbo1bit
./BitNet-2B.llamafile --cli -p "Hello"
```

### Supported 1-bit Models

| Model | Parameters | Source | Notes |
|-------|-----------|--------|-------|
| `prism-ml/Bonsai-8B-gguf` | 8B | HuggingFace | Pre-converted GGUF, no conversion needed |
| `prism-ml/Bonsai-1.7B-gguf` | 1.7B | HuggingFace | Pre-converted GGUF, no conversion needed |
| `microsoft/BitNet-b1.58-2B-4T` | 2B | HuggingFace | Official Microsoft model |
| `HF1BitLLM/Llama3-8B-1.58-100B-tokens` | 8B | HuggingFace | LLaMA-3 architecture, 100B training tokens |
| `tiiuae/Falcon3-7B-1.58bit` | 7B | HuggingFace | TII Falcon3 series |
| `tiiuae/Falcon3-10B-1.58bit` | 10B | HuggingFace | |
| `tiiuae/Falcon3-3B-1.58bit` | 3B | HuggingFace | |
| `tiiuae/Falcon3-1B-1.58bit` | 1B | HuggingFace | |
| `1bitLLM/bitnet_b1_58-large` | 700M | HuggingFace | Original 1bitLLM |
| `1bitLLM/bitnet_b1_58-3B` | 3B | HuggingFace | |

### Performance vs Standard Models

At 1K context on Apple M1 Pro:

| Model | Format | RAM (weights) | Speed |
|-------|--------|--------------|-------|
| LLaMA-3-8B | FP16 | 16 GB | ~25 tok/s |
| LLaMA-3-8B | Q4_0 | 5 GB | ~45 tok/s |
| BitNet-8B / Bonsai-8B | 1-bit | **~2 GB** | **~60 tok/s** |

At 65K context, only 1-bit models with Turbo1Bit remain practical — FP16 8B models
would require ~50 GB for the KV cache alone.

### Turbo1Bit: KV Cache Compression

For 1-bit models, llamafile automatically enables **Turbo1Bit** — 4-bit quantized
KV cache via Flash Attention. This is triggered by filenames containing "bonsai",
"bitnet", "1bit", or "prism".

```bash
# Auto-enabled for recognized filenames
llamafile -m Bonsai-8B.gguf -c 65536

# Explicitly enable for any model
llamafile -m model.gguf --turbo1bit -c 65536

# Disable auto-injection
llamafile -m Bonsai-8B.gguf --no-turbo1bit

# Manual equivalent of --turbo1bit
llamafile -m Bonsai-8B.gguf --fa --ctk q4_0 --ctv q4_0 -c 65536
```

KV cache memory at 65K context with 4-bit compression is **~2.65x smaller** than
the FP16 baseline, at less than 0.5% perplexity cost. Higher precision options:

| Flag combo | Compression | Quality impact |
|-----------|-------------|----------------|
| `--ctk q4_0 --ctv q4_0` | 2.65x | < 0.5% perplexity |
| `--ctk q5_0 --ctv q5_0` | 2.32x | < 0.3% perplexity |
| `--ctk q8_0 --ctv q8_0` | 1.68x | negligible |

All require `--fa` (Flash Attention). `--turbo1bit` injects `--fa --ctk q4_0 --ctv q4_0` automatically.

### BitNet Inference Kernels

This fork integrates [Microsoft BitNet.cpp](https://github.com/microsoft/BitNet)
LUT-based kernels that replace floating-point multiply-accumulate with integer table
lookups for ternary weights:

| Kernel | Platform | When used |
|--------|----------|-----------|
| TL1 | ARM64 (NEON) | Apple M-series, ARM servers — fastest |
| TL2 | x86_64 (AVX2) | Intel/AMD CPUs — fastest |
| I2_S | Universal | cosmocc portable build (default) |

The cosmocc build uses I2_S for maximum portability. For peak performance with TL1/TL2,
use the cmake build with clang ≥ 18:

```bash
# ARM64 (Apple Silicon, ARM servers)
cmake -B build -DBITNET_ARM_TL1=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release

# x86_64 (Intel/AMD)
cmake -B build -DBITNET_X86_TL2=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release
```

### Converting BitNet Models to GGUF

HuggingFace `.safetensors` checkpoints must be converted before use. The
`tools/bitnet/` scripts handle the full pipeline:

```bash
# Install Python conversion dependencies
uv sync --group convert

# Download + convert + quantize in one step
mise run model:convert -- models/bitnet-2B --hf-repo microsoft/BitNet-b1.58-2B-4T

# Or run the converter directly
uv run python tools/bitnet/setup_env.py \
  --hf-repo microsoft/BitNet-b1.58-2B-4T \
  --quant-type i2_s
```

Output GGUF formats:

| Format | Description | Platforms |
|--------|-------------|-----------|
| `i2_s` | Packed 2-bit signed | All (universal fallback) |
| `tl1` | ARM lookup table kernels | Apple M-series, ARM64 servers |
| `tl2` | x86 lookup table kernels | Intel/AMD |

### Architecture

The 1-bit support is built on three components:

1. **[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp/tree/prism) (prism branch)** — replaces the upstream llama.cpp submodule, adds Bonsai 1-bit weight format support and the I2_S/TL1/TL2 quantization types
2. **[Microsoft BitNet.cpp](https://github.com/microsoft/BitNet) kernels** — LUT-based inference kernels for ternary weights, integrated in `tools/bitnet/`
3. **[Turbo1Bit](docs/turbo1bit.md) KV cache compression** — auto-enabled for 1-bit models to bound memory at long contexts

> **Note:** Bonsai/BitNet models require the PrismML fork. The quantization types `I2_S`, `TL1`, and `TL2` are not supported by mainline ggerganov/llama.cpp without patches.

See [docs/bitnet.md](docs/bitnet.md) and [docs/turbo1bit.md](docs/turbo1bit.md) for full documentation.

---

## Execution Modes

llamafile supports four execution modes selected by a flag:

| Flag | Description |
|------|-------------|
| *(default)* | Combined: TUI chat in the foreground + HTTP server in the background |
| `--chat` | Interactive TUI chat only, model loaded directly into memory |
| `--server` | HTTP server only, exposes an OpenAI-compatible API |
| `--cli` | Single prompt → response, then exit — designed for scripting |

```sh
llamafile -m model.gguf                          # combined TUI + server
llamafile -m model.gguf --chat                   # TUI chat only
llamafile -m model.gguf --server --port 8080     # HTTP API only
llamafile -m model.gguf --cli -p "write a haiku" # one-shot response
```

### Common Options

```
-m FILE          path to GGUF model file (required)
-p TEXT          system prompt (in --cli mode: user prompt)
-c N             context window size in tokens
-n N             max tokens to generate
--temp N         sampling temperature (0 = deterministic)
--top-p N        top-p nucleus sampling
--top-k N        top-k sampling
--repeat-penalty N  repetition penalty
-ngl N           GPU layers to offload (-1 = all)
--gpu MODE       GPU backend: auto, nvidia, amd, apple, disable
--verbose        enable verbose logging
--version        print version and exit
--help           show help (combine with a mode flag for mode-specific help)
```

### CLI Mode (`--cli`)

Sends a single prompt and prints the response cleanly to stdout, then exits. No logo, no decorations — suitable for piping and scripting.

```sh
llamafile -m model.gguf --cli -p "summarize this: $(cat file.txt)"
llamafile -m model.gguf --cli --nothink -p "write a haiku"
llamafile -m model.gguf --cli --mmproj mm.gguf --image photo.jpg -p "describe this"
```

- `-p TEXT` is the **user prompt** in CLI mode (not the system prompt)
- `--nothink` strips `<think>…</think>` reasoning blocks from the output
- A system prompt can be set via config file (see [Configuration](#configuration)) or `--system-prompt` / `-sp`
- Multimodal: pass `--mmproj` and `--image` for vision models

### Chat Mode (`--chat`)

Interactive TUI with syntax-highlighted responses, conversation history, and slash commands.

```sh
llamafile -m model.gguf --chat
llamafile -m model.gguf --chat -p "You are a pirate"
llamafile -m model.gguf --chat --mmproj mmproj.gguf   # vision-capable
```

**Chat commands** (type at the `>>>` prompt):

| Command | Description |
|---------|-------------|
| `/help [CMD]` | Show all commands, or detailed help for one |
| `/clear` | Restart conversation from the beginning |
| `/context` | Show context window token usage |
| `/stats` | Show prompt and generation speed (tokens/sec) |
| `/dump [FILE]` | Print raw token history to terminal or save to file |
| `/upload FILE` | Share a text or image file with the assistant |
| `/push` / `/pop` | Save and restore conversation state (branch/backtrack) |
| `/stack` | Print the saved conversation stack |
| `/undo` | Erase the last exchange (question + answer) |
| `/forget` | Erase the oldest message (frees context, preserves system prompt) |
| `/manual [on\|off]` | Toggle manual role mode to inject system/assistant messages |
| `/exit` | Quit |

**Multi-line input:** use triple quotes `"""` to open a multi-line block, or press `Ctrl-J` to insert a newline without submitting.

### Server Mode (`--server`)

Runs an OpenAI-compatible HTTP API. See `llamafile --server --help` for the full list of server options.

```sh
llamafile -m model.gguf --server --port 8080 --host 0.0.0.0
```

### Combined Mode (default)

The default mode runs the HTTP server on the main thread and launches the TUI as an HTTP client on a background thread. This gives you both an interactive chat interface and an API simultaneously.

---

## Configuration

llamafile loads settings from `~/.config/llamafile/llamafile.yaml` (respects `$XDG_CONFIG_HOME`). Configuration is organized into sections — one per mode — with a `[global]` section that applies to all modes unless overridden.

**Command-line arguments always take precedence over config file values.**

### `--config` flag

```sh
# Print an annotated config template to stdout
llamafile --config

# Load settings from a specific file
llamafile -m model.gguf --config ~/myconfig.yaml
```

### Config file format

```yaml
# ~/.config/llamafile/llamafile.yaml

[global]
# Applies to all modes unless overridden in a mode section.

# Inline system prompt
system_prompt: You are a helpful assistant.

# Or load the system prompt from a file (supports ~/)
# system_prompt_file: ~/.config/llamafile/system_prompt.txt

# Sampling temperature. 0 = deterministic. Equivalent to --temp.
temp: 0.7

# Top-p nucleus sampling. Equivalent to --top-p.
top_p: 0.9

# Top-k sampling. 0 = disabled. Equivalent to --top-k.
top_k: 40

# Repetition penalty. 1.0 = disabled. Equivalent to --repeat-penalty.
repeat_penalty: 1.1

# Max tokens to generate. Equivalent to -n.
n_predict: 512

# Context window size. Equivalent to -c.
ctx_size: 4096

# GPU layers to offload. -1 = all. Equivalent to -ngl.
n_gpu_layers: 99

[cli]
# Settings for --cli mode.
# Note: system_prompt here sets the system message in the chat template.
# The user prompt is still provided via -p on the command line.
temp: 0.0
n_predict: 1024
system_prompt: You are a concise assistant. Answer briefly.

[chat]
# Settings for --chat mode (interactive TUI).
ctx_size: 8192
temp: 0.7

[server]
# Settings for --server mode.
ctx_size: 4096
n_gpu_layers: 99

[auto]
# Settings for the default combined mode (TUI + server).
ctx_size: 4096
```

**Supported keys** (all optional, all sections):

| Key | Aliases | Description |
|-----|---------|-------------|
| `system_prompt` | | Inline system prompt text |
| `system_prompt_file` | | Path to a file containing the system prompt |
| `temp` | `temperature` | Sampling temperature |
| `top_p` | | Top-p nucleus sampling |
| `top_k` | | Top-k sampling |
| `repeat_penalty` | | Repetition penalty |
| `n_predict` | `max_tokens` | Max tokens to generate |
| `ctx_size` | `context_size` | Context window size |
| `n_gpu_layers` | `gpu_layers` | GPU layers to offload |

Values in `[global]` act as defaults. Any key set in a mode section (e.g. `[cli]`) overrides the global value for that mode only.

### Development with mise

This fork uses [mise](https://mise.jdx.dev/) for development tooling:

```bash
# Install mise (see https://mise.jdx.dev/getting-started.html)
curl https://mise.run | sh

# Install tools + Python deps
mise install
uv sync

# Available tasks
mise run setup          # Initialize submodules and apply patches
mise run build          # Build llamafile
mise run test           # Run test suite
mise run clean          # Remove build artifacts
mise run model:download # Download Bonsai GGUF models
mise run model:convert  # Convert HF models to GGUF
mise run model:bundle   # Bundle GGUF into llamafile
mise run turbo1bit:build # Build standalone TurboQuant tools
mise run turbo1bit:bench # Benchmark KV cache compression
```

## Documentation

Check the full documentation in the [docs/](docs/) folder or online at [mozilla-ai.github.io/llamafile](https://mozilla-ai.github.io/llamafile/), or directly jump into one of the following subsections:

- [Quickstart](https://mozilla-ai.github.io/llamafile/quickstart/)
- [Example llamafiles](https://mozilla-ai.github.io/llamafile/example_llamafiles/)
- [Running a llamafile](https://mozilla-ai.github.io/llamafile/running_llamafile/)
- [Creating llamafiles](https://mozilla-ai.github.io/llamafile/creating_llamafiles/)
- [Source installation](https://mozilla-ai.github.io/llamafile/source_installation/)
- [Technical details](https://mozilla-ai.github.io/llamafile/technical_details/)
- [Supported Systems](https://mozilla-ai.github.io/llamafile/support/)
- [Troubleshooting](https://mozilla-ai.github.io/llamafile/troubleshooting/)
- [Whisperfile](https://mozilla-ai.github.io/llamafile/whisperfile/)


## Licensing

While the llamafile project is Apache 2.0-licensed, our changes
to llama.cpp and whisper.cpp are licensed under MIT (just like the projects
themselves) so as to remain compatible and upstreamable in the future,
should that be desired.

The llamafile logo on this page was generated with the assistance of DALL·E 3.


[![Star History Chart](https://api.star-history.com/svg?repos=Mozilla-Ocho/llamafile&type=Date)](https://star-history.com/#Mozilla-Ocho/llamafile&Date)
