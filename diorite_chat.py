"""
Diorite Chat - simple CLI chat, no server needed.
Usage: python diorite_chat.py
Type 'exit' / 'quit' to stop.
"""
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from diorite_config import BASE_MODEL, MODEL_NAME, OUTPUT_DIR, SYSTEM_PROMPT


def main():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    lora_path = Path(OUTPUT_DIR) / MODEL_NAME
    merged_path = Path(OUTPUT_DIR) / f"{MODEL_NAME}-merged"

    if lora_path.exists():
        from peft import PeftModel
        print(f"[+] Loading LoRA adapter from {lora_path}")
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(lora_path))
        tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
    elif merged_path.exists():
        print(f"[+] Loading merged model from {merged_path}")
        model = AutoModelForCausalLM.from_pretrained(str(merged_path), torch_dtype=dtype, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(str(merged_path))
    else:
        print(f"[!] No trained model found, loading base {BASE_MODEL}")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print("Diorite ready. Ask a CTF question (exit to quit).")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        prompt = f"{SYSTEM_PROMPT}\n\n[INST] {q}[/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, temperature=0.3, top_p=0.9, repetition_penalty=1.1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        print("\nDiorite:", tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
