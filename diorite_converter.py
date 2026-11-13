"""
Diorite Converter - Merges LoRA weights and exports to GGUF / ONNX

FIX vs original:
- GGUF and ONNX are now OPTIONAL (--skip-gguf / --skip-onnx).
- GGUF no longer fails silently when llama.cpp is missing. It checks for
  llama.cpp/convert_hf_to_gguf.py first, and if it is not there it saves
  the merged model and prints the exact manual command instead of crashing.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from diorite_config import (
    BASE_MODEL,
    GGUF_OUTPUT,
    MODEL_NAME,
    ONNX_OUTPUT,
    OUTPUT_DIR,
)


def merge_lora_adapter(base_model_name: str, lora_path: str, merged_path: str):
    """Merge LoRA adapter into base model and save full weights."""
    print(f"[+] Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print(f"[+] Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("[+] Merging LoRA weights into base model")
    model = model.merge_and_unload()

    print(f"[+] Saving merged model to: {merged_path}")
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(merged_path)

    total = sum(f.stat().st_size for f in Path(merged_path).rglob("*") if f.is_file())
    print(f"[+] Merged model saved. Total size: {total / (1024*1024):.1f} MB")
    return merged_path


def convert_to_gguf(merged_model_path: str, output_dir: str, quantization: str = "q4_0"):
    """Convert merged model to GGUF. Returns path or None if skipped / not possible."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / f"{MODEL_NAME}.gguf"

    convert_script = Path(__file__).parent / "llama.cpp" / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print("[!] llama.cpp convert script not found at ./llama.cpp/convert_hf_to_gguf.py")
        print("[!] Skipping GGUF - merged model is still usable as-is.")
        print("[!] To convert manually:")
        print("    git clone https://github.com/ggerganov/llama.cpp")
        print(f"    python llama.cpp/convert_hf_to_gguf.py {merged_model_path} --outfile {target} --outtype {quantization}")
        info_path = output_path / "conversion_info.json"
        with open(info_path, "w") as f:
            json.dump(
                {
                    "base_model": merged_model_path,
                    "format": "gguf",
                    "quantization": quantization,
                    "target_file": f"{MODEL_NAME}.gguf",
                    "manual_conversion_cmd": (
                        f"python llama.cpp/convert_hf_to_gguf.py {merged_model_path} "
                        f"--outfile {target} --outtype {quantization}"
                    ),
                },
                f,
                indent=2,
            )
        return None

    print(f"[+] Converting to GGUF ({quantization})")
    cmd = [
        sys.executable,
        str(convert_script),
        merged_model_path,
        "--outfile",
        str(target),
        "--outtype",
        quantization,
    ]
    subprocess.run(cmd, check=True)
    print(f"[+] GGUF model saved to: {target}")
    return str(target)


def convert_to_onnx(merged_model_path: str, output_dir: str):
    """Convert merged model to ONNX. Returns path or None on failure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print("[+] Converting to ONNX format")
    try:
        from optimum.onnxruntime import ORTModelForCausalLM

        model = ORTModelForCausalLM.from_pretrained(merged_model_path, export=True)
        model.save_pretrained(str(output_path))
        tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        tokenizer.save_pretrained(str(output_path))
        print(f"[+] ONNX model saved to: {output_path}")
        for f in output_path.glob("*.onnx"):
            print(f"    {f.name}: {f.stat().st_size / (1024*1024):.1f} MB")
        return str(output_path)
    except Exception as e:
        print(f"[!] ONNX export failed: {e}")
        print("[!] Manual steps:")
        print("    pip install optimum[onnxruntime]")
        print(f"    optimum-cli export onnx --model {merged_model_path} {output_path}")
        return None


def get_model_size(path: str) -> float:
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Diorite converter - merge LoRA, export GGUF/ONNX")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF export")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export")
    parser.add_argument("--quantization", default="q4_0", help="GGUF quantization (default: q4_0)")
    args = parser.parse_args()

    lora_path = Path(OUTPUT_DIR) / MODEL_NAME
    merged_path = Path(OUTPUT_DIR) / f"{MODEL_NAME}-merged"

    if not lora_path.exists():
        print(f"[!] LoRA adapter not found at: {lora_path}")
        print("[!] Run diorite_trainer.py first")
        sys.exit(1)

    print("=" * 60)
    print("  DIORITE CONVERTER - Model Format Conversion")
    print(f"  LoRA Adapter: {lora_path}")
    print(f"  Merged Output: {merged_path}")
    print(f"  GGUF Output: {GGUF_OUTPUT} (skip={args.skip_gguf})")
    print(f"  ONNX Output: {ONNX_OUTPUT} (skip={args.skip_onnx})")
    print("=" * 60)

    merged = merge_lora_adapter(BASE_MODEL, str(lora_path), str(merged_path))
    print(f"\n[+] Merged model size: {get_model_size(merged):.1f} MB")

    if not args.skip_gguf:
        print("\n" + "-" * 40)
        convert_to_gguf(merged, GGUF_OUTPUT, quantization=args.quantization)
    else:
        print("\n[-] GGUF export skipped")

    if not args.skip_onnx:
        print("\n" + "-" * 40)
        convert_to_onnx(merged, ONNX_OUTPUT)
    else:
        print("\n[-] ONNX export skipped")

    print("\n" + "=" * 60)
    print("  CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Merged HF model: {merged} ({get_model_size(merged):.1f} MB)")
    print("\n  Use diorite_server.py to serve the model")


if __name__ == "__main__":
    main()
