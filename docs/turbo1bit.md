# Turbo1Bit: KV Cache Compression for 1-bit LLMs

Turbo1Bit is a KV cache compression technique specifically designed for 1-bit LLMs
like Bonsai that dramatically reduces memory requirements at long context lengths.

## What Turbo1Bit Does

At long contexts (e.g., 65K tokens), the KV cache becomes the dominant memory
consumer — often larger than the model weights themselves. Turbo1Bit solves this
with two complementary compression techniques:

### 1. TurboQuant Key Compression

Keys are compressed using the **Prod quantizer** algorithm (Algorithm 2 from the
TurboQuant paper):

1. **Rotate**: Apply a random orthogonal matrix Pi (generated via QR decomposition)
   to decorrelate the key vector coordinates.
2. **MSE quantize**: Quantize each rotated coordinate using a pre-computed Lloyd-Max
   codebook at `(bits - 1)` bits per coordinate.
3. **QJL projection**: Project the residual (original - MSE reconstruction) through
   a Gaussian sketch matrix S, and store only the sign bits.

This yields inner-product-unbiased estimates of `<query, key>` scores — the quantity
needed for attention computation.

### 2. Group-Quantized Values

Values are compressed with simple min-max group quantization:
- Default: 4-bit quantization with group_size=32
- Memory reduction: 4x vs. FP16, with 0.998 cosine similarity

### 3. Recent-Token Buffer

A configurable number of recent tokens (default: 128) are kept at **full precision**
in a sliding buffer. Only tokens older than this threshold are compressed. This
preserves full quality for the local attention window while compressing the distant
context.

## Memory Savings

At 65K context length, head_dim=128, with 3-bit key + 4-bit value compression:

| Metric | Value |
|--------|-------|
| Key compression ratio | ~5.3x (3-bit Prod vs FP16) |
| Value compression ratio | ~4x (4-bit group quant) |
| Overall KV memory reduction | ~35–65% |
| Quality impact (perplexity) | <0.5% increase |

The savings are most dramatic at long contexts (>32K tokens) where the KV cache
dominates memory usage.

## Native KV Cache Quantization (Recommended)

For practical memory savings, the recommended approach uses llama.cpp's built-in
**native KV cache quantization** with Flash Attention:

```bash
# 4-bit KV cache — 2.65x memory reduction, requires Flash Attention
llamafile -m Bonsai-8B.gguf --fa --ctk q4_0 --ctv q4_0 -c 65536

# 5-bit KV cache — 2.32x memory reduction
llamafile -m Bonsai-8B.gguf --fa --ctk q5_0 --ctv q5_0 -c 65536

# 8-bit KV cache — 1.68x memory reduction
llamafile -m Bonsai-8B.gguf --fa --ctk q8_0 --ctv q8_0 -c 65536
```

This is what the `--turbo1bit` flag enables automatically.

## Quick Start

### Download and Run a Bonsai Model

```bash
# Download Bonsai GGUF
mise run model:download -- prism-ml/Bonsai-8B-gguf models/Bonsai-8B

# Run with Turbo1Bit (--fa + --ctk q4_0 + --ctv q4_0 auto-injected)
./o//llamafile/llamafile -m models/Bonsai-8B/Bonsai-8B.gguf --turbo1bit -c 65536

# Or run with explicit flags
./o//llamafile/llamafile \
  -m models/Bonsai-8B/Bonsai-8B.gguf \
  --fa --ctk q4_0 --ctv q4_0 \
  -c 65536
```

### Auto-Detection

For models with "bonsai", "bitnet", "1bit", or "prism" in their filename,
llamafile automatically enables Turbo1Bit flags:

```bash
# Auto-enabled (filename contains "bonsai")
llamafile -m Bonsai-8B.gguf -c 65536

# Explicitly disabled
llamafile -m Bonsai-8B.gguf --no-turbo1bit -c 65536
```

### Bundle a Bonsai model into a llamafile

```bash
# Bundle with Turbo1Bit defaults baked in
mise run model:bundle -- models/Bonsai-8B/Bonsai-8B.gguf Bonsai-8B.llamafile --turbo1bit

# Run the bundled llamafile (flags are embedded in .args)
./Bonsai-8B.llamafile --cli -p "Explain quantum computing"
./Bonsai-8B.llamafile  # starts TUI chat + HTTP server
```

## Using the turbo1bit-infer Tool

The `turbo1bit-infer` tool provides a reference implementation of TurboQuant
compression applied post-decode (simulating compressed KV storage for quality testing):

```bash
# Build standalone tools (requires cmake + clang >= 18)
mise run turbo1bit:build

# Run inference with Turbo1Bit compression simulation
./o/turbo1bit-build/turbo1bit-infer \
  -m models/Bonsai-8B/Bonsai-8B.gguf \
  -p "Write a poem about 1-bit neural networks" \
  --ctk q4_0 --ctv q4_0 --fa \
  -n 200

# Baseline (no compression) for quality comparison
./o/turbo1bit-build/turbo1bit-infer \
  -m models/Bonsai-8B/Bonsai-8B.gguf \
  -p "Write a poem about 1-bit neural networks" \
  --no-turbo1bit \
  -n 200
```

### turbo1bit-infer CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m FILE` | (required) | GGUF model file |
| `-p TEXT` | (required) | Input prompt |
| `-n N` | 128 | Tokens to generate |
| `-c N` | 2048 | Context size |
| `--fa` | off | Enable Flash Attention (required for --ctk/--ctv) |
| `--ctk TYPE` | f16 | Key cache type: f16, q8_0, q5_0, q4_0 |
| `--ctv TYPE` | f16 | Value cache type: f16, q8_0, q5_0, q4_0 |
| `--key-bits N` | 0 | TurboQuant key bits (0=disabled, 3-5 recommended) |
| `--val-bits N` | 4 | Value group quantization bits (2 or 4) |
| `--no-turbo1bit` | off | Disable in-place TurboQuant compression |

## Benchmarking

```bash
# Run memory and throughput benchmark suite
mise run turbo1bit:bench -- models/Bonsai-8B/Bonsai-8B.gguf 65536

# Manual benchmark with turbo1bit-bench
./o/turbo1bit-build/turbo1bit-bench
```

The benchmark measures:
- Memory usage at various context lengths (1K, 4K, 16K, 65K)
- KV cache compression ratios for different bit widths
- Attention score accuracy vs. FP16 baseline
- Prefill and decode throughput

## TurboQuant Algorithm Details

The TurboQuant library implements two quantizers:

### MSE Quantizer (Algorithm 1)

Optimizes squared error, suitable for value compression:

```
x → rotate(x, Pi) → Lloyd-Max quantize(bits) → bit-pack
```

### Prod Quantizer (Algorithm 2)

Optimizes inner product preservation, suitable for key compression:

```
x → [MSE at (bits-1)] + [QJL sign on residual]
```

The QJL (Quantized Johnson-Lindenstrauss) component adds 1 bit of inner product
information via the sign of a Gaussian sketch, achieving better attention score
accuracy than pure MSE at the same total bit budget.

### Codebooks

The Lloyd-Max codebooks embedded in `turbo1bit/src/turbo1bit_codebook.c` are
pre-computed for:
- `d ∈ {64, 128}` (head dimensions)
- `bits ∈ {1, 2, 3, 4}`

These are optimal scalar quantizers for the Beta distribution arising from
random rotation of unit-norm vectors.

## Source Code

The TurboQuant library is vendored from [jhammant/Turbo1bit](https://github.com/jhammant/Turbo1bit):

```
turbo1bit/
├── src/
│   ├── turbo1bit_codebook.h/c      Lloyd-Max codebooks
│   ├── turbo1bit_rotation.h/c      QR rotation + QJL projection
│   ├── turbo1bit_quantizer.h/c     MSE + Prod quantizers
│   ├── turbo1bit_kv_cache.h/c      Compressed KV cache
│   ├── turbo1bit_metal.h/m         Metal GPU (Apple Silicon)
│   ├── turbo1bit_metal.metal       Metal compute shaders
│   └── llama-kv-cache-accessors.h  llama.cpp KV cache integration
├── tools/turbo1bit/
│   ├── turbo1bit_infer.cpp         End-to-end inference tool
│   ├── turbo1bit_bench.c           Benchmark suite
│   └── turbo1bit_stress.c          Stress testing
├── BUILD.mk                        cosmocc build rules
└── CMakeLists.txt                  CMake build (standalone tools)
```

## References

- [Turbo1Bit GitHub](https://github.com/jhammant/Turbo1bit)
- [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp/tree/prism)
- [BitNet.cpp](https://github.com/microsoft/BitNet)
- [KVSharer](https://arxiv.org/abs/2405.16726) — related KV sharing work
- [KIVI](https://arxiv.org/abs/2402.02750) — per-channel KV quantization
- [QuIP#](https://arxiv.org/abs/2402.04396) — random rotation for quantization
