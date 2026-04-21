# BitNet: 1-bit LLM Support in llamafile

This llamafile fork adds native support for running **1-bit LLMs** (BitNet-style models)
via the PrismML llama.cpp fork and Microsoft's BitNet.cpp optimized inference kernels.

## What Is BitNet?

BitNet is a family of 1-bit (or 1.58-bit) large language models where weights are
quantized to `{-1, 0, +1}` (ternary). This dramatically reduces:

- **Model weights memory**: ~16x smaller than FP16 equivalents
- **Inference compute**: Integer arithmetic replaces FP multiply-accumulate
- **Energy consumption**: Estimated 10–100x reduction on specialized hardware

Notable BitNet models:
- `microsoft/BitNet-b1.58-2B-4T` — 2B parameters, trained on 4T tokens
- `1bitLLM/bitnet_b1_58-large` — LLaMA-architecture, 7B-class quality
- `HF1BitLLM/Llama3-8B-1.58-100B-tokens` — LLaMA-3-based, 100B training tokens
- `tiiuae/Falcon3-7B-1.58bit` — Falcon3 1.58-bit series
- `prism-ml/Bonsai-8B-gguf` — Bonsai 1-bit model (PrismML-Eng fork required)

## Architecture

This fork uses:

1. **[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (prism branch)**
   — A fork of llama.cpp with Bonsai 1-bit weight format support. This replaces
   the upstream ggerganov/llama.cpp submodule.

2. **[Microsoft BitNet.cpp](https://github.com/microsoft/BitNet) kernels**
   — Optimized LUT-based and MAD-based inference kernels for 1-bit weights.
   Architecture-specific:
   - **ARM64**: TL1 (lookup table) kernels
   - **x86_64**: TL2 (lookup table) kernels
   - **Fallback**: I2_S (packed 2-bit signed) format

3. **[Turbo1Bit](turbo1bit.md) KV cache compression**
   — Automatically enabled for Bonsai/BitNet models to reduce KV cache memory.

## Quick Start

### Running a BitNet Model

```bash
# Download Microsoft's BitNet-b1.58-2B-4T
mise run model:download -- microsoft/BitNet-b1.58-2B-4T models/bitnet-2B

# Convert to GGUF (if not pre-converted)
mise run model:convert -- models/bitnet-2B

# Run with llamafile
./o//llamafile/llamafile -m models/bitnet-2B/ggml-model-i2_s.gguf --turbo1bit

# Run in CLI mode
./o//llamafile/llamafile \
  -m models/bitnet-2B/ggml-model-i2_s.gguf \
  --cli -p "Explain 1-bit neural networks" \
  --fa --ctk q4_0 --ctv q4_0
```

### Downloading Models

```bash
# Download Bonsai-8B GGUF (ready to use, no conversion needed)
mise run model:download -- prism-ml/Bonsai-8B-gguf models/Bonsai-8B

# Download BitNet-b1.58-2B-4T (HuggingFace safetensors, needs conversion)
mise run model:download -- microsoft/BitNet-b1.58-2B-4T models/bitnet-2B

# Download Falcon3 1.58-bit
mise run model:download -- tiiuae/Falcon3-7B-1.58bit models/Falcon3-7B-1.58bit
```

### Converting Models to GGUF

BitNet models distributed as HuggingFace checkpoints (`.safetensors`) must be
converted to GGUF before use with llamafile:

```bash
# Full pipeline: download, convert, quantize
mise run model:convert -- models/bitnet-2B --hf-repo microsoft/BitNet-b1.58-2B-4T

# Manual conversion using the helper script
uv run python tools/bitnet/setup_env.py \
  --hf-repo microsoft/BitNet-b1.58-2B-4T \
  --quant-type i2_s

# Convert a local model directory
uv run python tools/bitnet/convert-helper-bitnet.py \
  models/bitnet-2B \
  --outtype i2_s
```

**Output formats:**

| Format | Description | Use when |
|--------|-------------|----------|
| `i2_s` | Packed 2-bit signed (universal) | All platforms |
| `tl1`  | ARM TL1 lookup table kernels | Apple M-series, ARM servers |
| `tl2`  | x86 TL2 lookup table kernels | Intel/AMD CPUs |
| `f32`  | Full precision (intermediate) | Conversion step only |

### Bundling a BitNet Model as a llamafile

```bash
# Build llamafile binary
mise run build

# Bundle with optimal 1-bit inference flags
mise run model:bundle -- models/bitnet-2B/ggml-model-i2_s.gguf BitNet-2B.llamafile --turbo1bit

# Run anywhere (no dependencies)
./BitNet-2B.llamafile --cli -p "Hello, world!"
./BitNet-2B.llamafile          # TUI chat + HTTP server
```

## Supported Models

All models supported by the BitNet.cpp setup_env.py pipeline:

| HF Repo | Parameters | Format | Notes |
|---------|-----------|--------|-------|
| `microsoft/BitNet-b1.58-2B-4T` | 2B | I2_S/TL1/TL2 | Official Microsoft model |
| `1bitLLM/bitnet_b1_58-large` | 700M | I2_S/TL1/TL2 | Original 1bitLLM |
| `1bitLLM/bitnet_b1_58-3B` | 3B | I2_S/TL1/TL2 | 3B variant |
| `HF1BitLLM/Llama3-8B-1.58-100B-tokens` | 8B | I2_S/TL1/TL2 | LLaMA-3 architecture |
| `tiiuae/Falcon3-7B-1.58bit` | 7B | I2_S/TL1/TL2 | TII Falcon3 series |
| `tiiuae/Falcon3-10B-1.58bit` | 10B | I2_S/TL1/TL2 | |
| `tiiuae/Falcon3-3B-1.58bit` | 3B | I2_S/TL1/TL2 | |
| `tiiuae/Falcon3-1B-1.58bit` | 1B | I2_S/TL1/TL2 | |
| `prism-ml/Bonsai-8B-gguf` | 8B | GGUF | PrismML, pre-converted |
| `prism-ml/Bonsai-1.7B-gguf` | 1.7B | GGUF | PrismML, pre-converted |

## BitNet Kernel Backends

### How BitNet.cpp Kernels Work

BitNet's optimized kernels use **lookup tables (LUT)** to replace floating-point
multiply-accumulate with integer table lookups:

```
Traditional: y += w * x      (1 FP multiply + 1 FP add)
BitNet LUT:  y += LUT[w_idx] (1 table lookup + 1 FP add)
```

For ternary weights `{-1, 0, +1}`, this achieves ~3x speedup on CPU without
specialized hardware.

### Kernel Variants

| Kernel | Platform | Quantization | Speed |
|--------|----------|-------------|-------|
| TL1 | ARM64 (NEON) | TL1 | Fastest on Apple M-series |
| TL2 | x86_64 (AVX2) | TL2 | Fastest on Intel/AMD |
| I2_S | Universal | 2-bit signed | Portable fallback |

### Building with BitNet Kernels

```bash
# ARM64 with TL1 kernels
cmake -B build -DBITNET_ARM_TL1=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release

# x86_64 with TL2 kernels
cmake -B build -DBITNET_X86_TL2=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release
```

With llamafile's cosmocc build, the I2_S fallback is used (maximum portability).
For peak BitNet performance, use the cmake build with clang ≥18 and platform
kernels enabled.

## Python Utilities

The `tools/bitnet/` directory contains Python utilities for model management:

```
tools/bitnet/
├── setup_env.py              Full model preparation pipeline
├── convert-helper-bitnet.py  HF → GGUF conversion helper
├── convert-hf-to-gguf-bitnet.py  BitNet-specific GGUF converter
├── codegen_tl1.py            ARM TL1 kernel code generation
└── codegen_tl2.py            x86 TL2 kernel code generation
```

All scripts use `uv` for dependency management:

```bash
# Install Python dependencies
uv sync --group convert

# Run any tool
uv run python tools/bitnet/setup_env.py --help
```

## Performance Comparison

At 1K context, comparing standard FP16 llama.cpp vs. BitNet I2_S:

| Model | Format | RAM (model) | Tokens/sec (M1 Pro) |
|-------|--------|-------------|---------------------|
| LLaMA-3-8B | FP16 | 16 GB | ~25 tok/s |
| LLaMA-3-8B | Q4_0 | 5 GB | ~45 tok/s |
| BitNet-8B-1.58bit | I2_S | 2 GB | ~60 tok/s* |
| Bonsai-8B | GGUF | 2 GB | ~60 tok/s* |

*With Turbo1Bit KV cache compression at 65K context, RAM stays bounded while
FP16 models would OOM.

## Troubleshooting

### Model loads but produces garbage output

This usually means the GGUF was built for a different llama.cpp version. Bonsai
models require the PrismML fork's GGUF format support. Ensure you're using the
`prism` branch of llama.cpp (this fork has it configured in `.gitmodules`).

### "GGML_ASSERT: tensor type not supported" errors

The BitNet quantization types (I2_S) require the PrismML fork. Mainline
ggerganov/llama.cpp does not support these types without patches.

### Slow inference despite TL1/TL2 kernels

Make sure clang ≥18 is used for compilation (older compilers won't emit the
required NEON/AVX2 intrinsics). Check with:

```bash
clang --version  # should show 18.x or higher
```

## References

- [BitNet paper](https://arxiv.org/abs/2310.11453) — "BitNet: Scaling 1-bit Transformers for Large Language Models"
- [BitNet b1.58 paper](https://arxiv.org/abs/2402.17764) — "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
- [BitNet.cpp](https://github.com/microsoft/BitNet) — Official Microsoft BitNet inference
- [PrismML llama.cpp](https://github.com/PrismML-Eng/llama.cpp/tree/prism) — Fork with Bonsai support
- [Turbo1Bit](turbo1bit.md) — KV cache compression for 1-bit models
