#!/usr/bin/env python3
"""
tools/bitnet/convert-helper-bitnet.py — Helper to convert BitNet HuggingFace
checkpoints (.safetensors) to GGUF format for use with llamafile.

This wraps the GGUF conversion utilities from llama.cpp's gguf-py library,
extended with BitNet-specific quantization type handling (I2_S, TL1, TL2).

Usage:
    uv run python tools/bitnet/convert-helper-bitnet.py <model_dir> [--outtype f32]
    uv run python tools/bitnet/convert-helper-bitnet.py ./models/bitnet-2B --outtype i2_s

This script requires the model directory to contain HuggingFace model files:
    config.json, tokenizer_config.json, model.safetensors (or pytorch_model.bin)

Output: <model_dir>/ggml-model-<outtype>.gguf

Source: Adapted from https://github.com/microsoft/BitNet/blob/main/utils/convert-helper-bitnet.py
License: MIT
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


SUPPORTED_OUTTYPES = ["f32", "f16", "i2_s", "tl1", "tl2"]

# Mapping from llamafile/BitNet types to llama.cpp convert_hf_to_gguf types
OUTTYPE_ALIASES = {
    "f32":  "f32",
    "f16":  "f16",
    # BitNet-specific: i2_s requires f32 intermediate then llama-quantize
    "i2_s": "f32",
    # TL1/TL2 are handled by convert-hf-to-gguf-bitnet.py with --outtype tl1/tl2
    "tl1":  "tl1",
    "tl2":  "tl2",
}


def find_conversion_script():
    """Find the convert-hf-to-gguf-bitnet.py script."""
    candidates = [
        Path(__file__).parent / "convert-hf-to-gguf-bitnet.py",
        Path("llama.cpp/tools/convert-hf-to-gguf.py"),
        Path("llama.cpp/convert_hf_to_gguf.py"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def find_quantize_bin():
    """Find llama-quantize binary."""
    candidates = [
        "o//llama.cpp/tools/quantize/llama-quantize",
        "o/rel/llama.cpp/tools/quantize/llama-quantize",
        "build/bin/llama-quantize",
        "llama.cpp/build/bin/llama-quantize",
    ]
    import shutil
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return shutil.which("llama-quantize")


def run(cmd):
    logger.info(f"Running: {' '.join(str(x) for x in cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        sys.exit(1)


def convert(model_dir: str, outtype: str, quant_embd: bool):
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    ggml_outtype = OUTTYPE_ALIASES[outtype]
    f32_path = os.path.join(model_dir, "ggml-model-f32.gguf")
    out_path = os.path.join(model_dir, f"ggml-model-{outtype}.gguf")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        logger.info(f"Output already exists: {out_path}")
        return out_path

    convert_script = find_conversion_script()
    if not convert_script:
        logger.error(
            "Conversion script not found. Ensure the llama.cpp submodule is "
            "initialized (mise run setup) or that convert-hf-to-gguf-bitnet.py "
            "is present in tools/bitnet/."
        )
        sys.exit(1)

    if outtype == "f16":
        # Convert directly to f16 GGUF
        logger.info(f"Converting {model_dir} to f16 GGUF...")
        run([sys.executable, convert_script, model_dir, "--outtype", "f16"])
        return out_path

    # Step 1: Convert to f32 GGUF (intermediate for quantization)
    logger.info(f"Step 1: Converting {model_dir} to f32 GGUF...")
    run([sys.executable, convert_script, model_dir, "--outtype", "f32"])

    if outtype == "f32":
        return f32_path

    # Step 2: Quantize to i2_s / tl1 / tl2
    logger.info(f"Step 2: Quantizing to {outtype}...")
    quantize_bin = find_quantize_bin()
    if not quantize_bin:
        logger.error(
            "llama-quantize binary not found. Run 'mise run build' first, "
            "or install llama.cpp with cmake."
        )
        sys.exit(1)

    if outtype == "i2_s":
        cmd = [quantize_bin]
        if quant_embd:
            cmd += ["--token-embedding-type", "f16"]
        cmd += [f32_path, out_path, "I2_S", "1"]
        if quant_embd:
            cmd.append("1")
        run(cmd)
    else:
        # tl1 / tl2 handled by BitNet-specific script with --outtype
        run([sys.executable, convert_script, model_dir, "--outtype", outtype])

    logger.info(f"Output: {out_path}")
    return out_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace BitNet model to GGUF format."
    )
    parser.add_argument("model_dir", help="Path to HuggingFace model directory")
    parser.add_argument(
        "--outtype",
        default="f32",
        choices=SUPPORTED_OUTTYPES,
        help="Output quantization type (default: f32)",
    )
    parser.add_argument(
        "--quant-embd",
        action="store_true",
        help="Quantize embeddings to f16",
    )
    args = parser.parse_args()

    out = convert(args.model_dir, args.outtype, args.quant_embd)
    print(f"Converted model: {out}")


if __name__ == "__main__":
    main()
