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

## 1-bit LLMs: Bonsai + BitNet + Turbo1Bit

This fork adds native support for **1-bit LLMs** (Bonsai, BitNet) via the
[PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp/tree/prism),
with automatic **Turbo1Bit KV cache compression** for efficient long-context inference.

### Run a Bonsai Model

```bash
# Download Bonsai-8B GGUF (pre-converted, no setup needed)
mise run model:download -- prism-ml/Bonsai-8B-gguf models/Bonsai-8B

# Run with Turbo1Bit compression auto-enabled (--fa --ctk q4_0 --ctv q4_0)
./llamafile -m models/Bonsai-8B/Bonsai-8B.gguf

# Explicit Turbo1Bit flags at 65K context
./llamafile -m models/Bonsai-8B/Bonsai-8B.gguf --fa --ctk q4_0 --ctv q4_0 -c 65536
```

For Bonsai and BitNet models (filenames containing "bonsai", "bitnet", "1bit"),
llamafile **automatically injects** `--fa --ctk q4_0 --ctv q4_0` for optimally
efficient inference. Disable with `--no-turbo1bit`.

### Run a Microsoft BitNet Model

```bash
# Download and convert BitNet-b1.58-2B-4T
mise run model:convert -- models/bitnet-2B --hf-repo microsoft/BitNet-b1.58-2B-4T

# Bundle as a standalone llamafile
mise run model:bundle -- models/bitnet-2B/ggml-model-i2_s.gguf BitNet-2B.llamafile --turbo1bit

# Run anywhere
./BitNet-2B.llamafile --cli -p "Explain 1-bit neural networks"
```

**Memory savings with Turbo1Bit at 65K context:**
- KV cache: ~2.65x smaller with 4-bit quantization
- Bonsai-8B fits comfortably in 16GB RAM at 65K context

See [docs/turbo1bit.md](docs/turbo1bit.md) and [docs/bitnet.md](docs/bitnet.md) for full documentation.

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
