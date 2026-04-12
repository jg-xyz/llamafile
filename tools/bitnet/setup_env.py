#!/usr/bin/env python3
"""
tools/bitnet/setup_env.py — BitNet model preparation pipeline for llamafile.

Adapted from microsoft/BitNet's setup_env.py to work with the llamafile
build system (cosmocc) and the Turbo1Bit-enhanced PrismML llama.cpp fork.

Usage:
    uv run python tools/bitnet/setup_env.py --model-dir models/BitNet-b1.58-2B-4T
    uv run python tools/bitnet/setup_env.py --hf-repo microsoft/BitNet-b1.58-2B-4T
    uv run python tools/bitnet/setup_env.py --model-dir models/Bonsai-8B-gguf --quant-type i2_s

For integration with mise tasks, use:
    mise run model:convert -- <model_dir>
    mise run model:download

Source: https://github.com/microsoft/BitNet/blob/main/setup_env.py
License: MIT
"""

import argparse
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("setup_env")

# ---------------------------------------------------------------------------
# Supported models
# ---------------------------------------------------------------------------

SUPPORTED_HF_MODELS = {
    # Microsoft official BitNet models
    "microsoft/BitNet-b1.58-2B-4T": {
        "model_name": "BitNet-b1.58-2B-4T",
    },
    # 1bitLLM community models
    "1bitLLM/bitnet_b1_58-large": {
        "model_name": "bitnet_b1_58-large",
    },
    "1bitLLM/bitnet_b1_58-3B": {
        "model_name": "bitnet_b1_58-3B",
    },
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens": {
        "model_name": "Llama3-8B-1.58-100B-tokens",
    },
    # TII Falcon3 1.58-bit models
    "tiiuae/Falcon3-7B-Instruct-1.58bit": {"model_name": "Falcon3-7B-Instruct-1.58bit"},
    "tiiuae/Falcon3-7B-1.58bit":          {"model_name": "Falcon3-7B-1.58bit"},
    "tiiuae/Falcon3-10B-Instruct-1.58bit":{"model_name": "Falcon3-10B-Instruct-1.58bit"},
    "tiiuae/Falcon3-10B-1.58bit":         {"model_name": "Falcon3-10B-1.58bit"},
    "tiiuae/Falcon3-3B-Instruct-1.58bit": {"model_name": "Falcon3-3B-Instruct-1.58bit"},
    "tiiuae/Falcon3-3B-1.58bit":          {"model_name": "Falcon3-3B-1.58bit"},
    "tiiuae/Falcon3-1B-Instruct-1.58bit": {"model_name": "Falcon3-1B-Instruct-1.58bit"},
    # PrismML Bonsai 1-bit models
    "prism-ml/Bonsai-8B-gguf":  {"model_name": "Bonsai-8B"},
    "prism-ml/Bonsai-1.7B-gguf":{"model_name": "Bonsai-1.7B"},
}

SUPPORTED_QUANT_TYPES = {
    "arm64": ["i2_s", "tl1"],
    "x86_64": ["i2_s", "tl2"],
}

COMPILER_EXTRA_ARGS = {
    "arm64":  ["-DBITNET_ARM_TL1=OFF"],
    "x86_64": ["-DBITNET_X86_TL2=OFF"],
}

OS_EXTRA_ARGS = {
    "Windows": ["-T", "ClangCL"],
}

ARCH_ALIAS = {
    "AMD64": "x86_64", "x86": "x86_64", "x86_64": "x86_64",
    "aarch64": "arm64", "arm64": "arm64", "ARM64": "arm64",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def system_info():
    return platform.system(), ARCH_ALIAS[platform.machine()]


def get_model_name(args):
    if args.hf_repo:
        return SUPPORTED_HF_MODELS[args.hf_repo]["model_name"]
    return os.path.basename(os.path.normpath(args.model_dir))


def run_command(command, shell=False, log_step=None, log_dir="logs"):
    if log_step:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(log_dir, log_step + ".log")
        with open(log_file, "w") as f:
            try:
                subprocess.run(command, shell=shell, check=True, stdout=f, stderr=f)
            except subprocess.CalledProcessError as e:
                logging.error(
                    f"Error while running command: {e}, see {log_file}"
                )
                sys.exit(1)
    else:
        try:
            subprocess.run(command, shell=shell, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error while running command: {e}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def prepare_model(args):
    _, arch = system_info()
    hf_url = args.hf_repo
    model_dir = args.model_dir
    quant_type = args.quant_type

    if hf_url is not None:
        model_dir = os.path.join(model_dir, SUPPORTED_HF_MODELS[hf_url]["model_name"])
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading model {hf_url} from HuggingFace to {model_dir}...")
        run_command(
            ["huggingface-cli", "download", hf_url, "--local-dir", model_dir],
            log_step="download_model",
            log_dir=args.log_dir,
        )
    elif not os.path.exists(model_dir):
        logging.error(f"Model directory {model_dir} does not exist.")
        sys.exit(1)
    else:
        logging.info(f"Loading model from directory {model_dir}.")

    gguf_path = os.path.join(model_dir, "ggml-model-" + quant_type + ".gguf")

    # Check if a pre-converted GGUF already exists (e.g., Bonsai distributes GGUF directly)
    existing_gguf = [
        f for f in Path(model_dir).glob("*.gguf")
        if "ggml-model" not in f.name
    ]
    if existing_gguf and not os.path.exists(gguf_path):
        logging.info(f"Found existing GGUF: {existing_gguf[0]} — skipping conversion.")
        return

    if not os.path.exists(gguf_path) or os.path.getsize(gguf_path) == 0:
        logging.info("Converting HF model to GGUF format...")

        convert_script = Path(__file__).parent / "convert-hf-to-gguf-bitnet.py"
        if not convert_script.exists():
            logging.warning(
                f"Conversion script not found: {convert_script}\n"
                "Skipping GGUF conversion. If your model is already in GGUF format "
                "this is expected."
            )
            return

        if quant_type.startswith("tl"):
            run_command(
                [sys.executable, str(convert_script), model_dir,
                 "--outtype", quant_type, "--quant-embd"],
                log_step="convert_to_tl",
                log_dir=args.log_dir,
            )
        else:  # i2_s
            f32_model = os.path.join(model_dir, "ggml-model-f32.gguf")
            i2s_model = os.path.join(model_dir, "ggml-model-i2_s.gguf")

            run_command(
                [sys.executable, str(convert_script), model_dir, "--outtype", "f32"],
                log_step="convert_to_f32_gguf",
                log_dir=args.log_dir,
            )

            # Locate quantize binary (llamafile build output)
            quantize_bin = _find_quantize_bin()
            if not quantize_bin:
                logging.error(
                    "llama-quantize binary not found. Run 'mise run build' first."
                )
                sys.exit(1)

            if args.quant_embd:
                run_command(
                    [quantize_bin, "--token-embedding-type", "f16",
                     f32_model, i2s_model, "I2_S", "1", "1"],
                    log_step="quantize_to_i2s",
                    log_dir=args.log_dir,
                )
            else:
                run_command(
                    [quantize_bin, f32_model, i2s_model, "I2_S", "1"],
                    log_step="quantize_to_i2s",
                    log_dir=args.log_dir,
                )

        logging.info(f"GGUF model saved at {gguf_path}")
    else:
        logging.info(f"GGUF model already exists at {gguf_path}")


def _find_quantize_bin():
    """Locate llama-quantize binary in llamafile build output."""
    candidates = [
        "o//llama.cpp/tools/quantize/llama-quantize",
        "o/rel/llama.cpp/tools/quantize/llama-quantize",
        "build/bin/llama-quantize",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return shutil.which("llama-quantize")


def gen_code(args):
    _, arch = system_info()
    model_name = get_model_name(args)

    llama3_f3_models = {
        m["model_name"] for m in SUPPORTED_HF_MODELS.values()
        if m["model_name"].startswith("Falcon") or m["model_name"].startswith("Llama")
    }

    codegen_script_dir = Path(__file__).parent
    tl1_script = codegen_script_dir / "codegen_tl1.py"
    tl2_script = codegen_script_dir / "codegen_tl2.py"

    if arch == "arm64":
        if not tl1_script.exists():
            logging.info("codegen_tl1.py not found — skipping kernel codegen.")
            return
        if model_name in ("bitnet_b1_58-large",):
            bm, bk, bm2 = "256,128,256", "128,64,128", "32,64,32"
        elif model_name in llama3_f3_models:
            bm, bk, bm2 = "256,128,256,128", "128,64,128,64", "32,64,32,64"
        elif model_name in ("bitnet_b1_58-3B", "BitNet-b1.58-2B-4T"):
            bm, bk, bm2 = "160,320,320", "64,128,64", "32,64,32"
        else:
            logging.info(f"No pre-tuned kernel for {model_name} — skipping codegen.")
            return
        run_command(
            [sys.executable, str(tl1_script), "--model", model_name,
             "--BM", bm, "--BK", bk, "--bm", bm2],
            log_step="codegen",
            log_dir=args.log_dir,
        )
    else:
        if not tl2_script.exists():
            logging.info("codegen_tl2.py not found — skipping kernel codegen.")
            return
        if model_name in ("bitnet_b1_58-large",):
            bm, bk, bm2 = "256,128,256", "96,192,96", "32,32,32"
        elif model_name in llama3_f3_models:
            bm, bk, bm2 = "256,128,256,128", "96,96,96,96", "32,32,32,32"
        elif model_name in ("bitnet_b1_58-3B", "BitNet-b1.58-2B-4T"):
            bm, bk, bm2 = "160,320,320", "96,96,96", "32,32,32"
        else:
            logging.info(f"No pre-tuned kernel for {model_name} — skipping codegen.")
            return
        run_command(
            [sys.executable, str(tl2_script), "--model", model_name,
             "--BM", bm, "--BK", bk, "--bm", bm2],
            log_step="codegen",
            log_dir=args.log_dir,
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    _, arch = system_info()
    parser = argparse.ArgumentParser(
        description="Prepare a BitNet / Bonsai model for llamafile inference."
    )
    parser.add_argument(
        "--hf-repo", "-hr",
        type=str,
        choices=list(SUPPORTED_HF_MODELS.keys()),
        help="HuggingFace model repo to download and convert",
    )
    parser.add_argument(
        "--model-dir", "-md",
        type=str,
        default="models",
        help="Directory to save/load the model",
    )
    parser.add_argument(
        "--log-dir", "-ld",
        type=str,
        default="logs",
        help="Directory to save log files",
    )
    parser.add_argument(
        "--quant-type", "-q",
        type=str,
        choices=SUPPORTED_QUANT_TYPES.get(arch, ["i2_s"]),
        default="i2_s",
        help="Quantization type for model conversion",
    )
    parser.add_argument(
        "--quant-embd",
        action="store_true",
        help="Quantize embedding layers to f16",
    )
    parser.add_argument(
        "--skip-codegen",
        action="store_true",
        help="Skip platform-specific kernel code generation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("=== BitNet Model Setup for llamafile ===")
    prepare_model(args)

    if not args.skip_codegen:
        gen_code(args)

    logging.info("Setup complete.")
    logging.info(
        "Next steps:\n"
        "  1. mise run model:bundle -- <model.gguf> [output.llamafile] [--turbo1bit]\n"
        "  2. ./output.llamafile --cli -p 'Your prompt'\n"
    )


def _signal_handler(sig, frame):
    logging.info("Interrupted.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    main()
