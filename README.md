# Diorite — a tiny CTF & cybersecurity LLM

> **SmolLM2-135M, LoRA fine-tuned for CTFs. Runs on CPU. Serves itself as a FastAPI.**
> Built for learning, labs, and CTF practice — not for attacking real systems.

Diorite turns HuggingFaceTB's **SmolLM2-135M-Instruct** into a focused assistant for binary exploitation, reverse engineering, web exploitation, cryptography, forensics, and Linux privilege escalation. It answers in a consistent format — **Analysis → Approach → Solution** — with runnable code snippets.

At 135M parameters, it's small enough to train and run on a laptop, which is the whole point: a complete, understandable fine-tuning pipeline you can read end-to-end in an afternoon.

## Why Diorite?

- **Tiny by design:** 135M base model, LoRA `r=16, alpha=32` — trainable on CPU, fast on GPU
- **Focused, not generic:** CTF-only system prompt + curated CTF dataset, no generic chatbot bloat
- **Complete pipeline:** dataset → LoRA training → merge → GGUF / ONNX export → FastAPI server
- **Built for integration:** REST + streaming endpoints, designed to plug into tools like JARVIS
- **Readable code:** 6 Python files, no hidden magic, every config in one place

## Quickstart

### 1. Install

```bash
git clone https://github.com/Nicogaming1/Diorite
cd diorite
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch. CPU works, GPU is faster.

### 2. Train

```bash
python diorite_trainer.py
```

This builds the dataset, applies LoRA, trains for 2 epochs, and saves the adapter to `./diorite-output/Diorite-135M-CTF/`.

By default it uses 200 examples on CPU and 2000 on GPU. Override with:

```bash
DIORITE_MAX_EXAMPLES=1000 python diorite_trainer.py
```

Training defaults (`diorite_config.py`): `lr=2e-4`, cosine schedule, `batch=1`, `grad_accum=2`, `max_seq_length=256`.

### 3. Convert (optional, for deployment)

```bash
python diorite_converter.py
```

Merges LoRA into the base model, then exports:
- Merged HF model → `./diorite-output/Diorite-135M-CTF-merged/`
- GGUF `q4_0` → `./diorite-gguf/` (needs llama.cpp for the GGUF step)
- ONNX → `./diorite-onnx/`

### 4. Serve

```bash
python diorite_server.py
```

Server runs on `http://127.0.0.1:8190`

```bash
# Health check
curl http://127.0.0.1:8190/health

# Ask a question
curl -X POST http://127.0.0.1:8190/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain how to exploit a basic buffer overflow.", "max_tokens": 512}'
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info + endpoints |
| `/health` | GET | Status, model loaded, uptime |
| `/generate` | POST | Generate a response |
| `/generate/stream` | POST | Stream tokens (SSE) |
| `/chat` | POST | Chat-style endpoint |

`POST /generate` body:

```json
{
  "prompt": "How do I crack a weak RSA key?",
  "max_tokens": 512,
  "temperature": 0.3,
  "top_p": 0.9
}
```

Response:

```json
{
  "response": "## Cracking Small RSA\n\n### Analysis\n...",
  "tokens_generated": 214,
  "latency_ms": 842.1,
  "model": "Diorite-135M-CTF",
  "tokens_per_second": 254.1
}
```

## Project structure

```
diorite/
├── diorite_config.py     # Base model, LoRA, training, server + system prompt
├── diorite_dataset.py    # Curated CTF examples + HF dataset loading
├── diorite_trainer.py    # LoRA fine-tuning with HF Trainer
├── diorite_converter.py  # Merge LoRA, export GGUF / ONNX
├── diorite_server.py     # FastAPI inference server (port 8190)
└── requirements.txt      # torch, transformers, peft, trl, fastapi, etc.
```

## What it covers

- **Binary exploitation:** buffer overflows, ret2libc, ROP basics
- **Reverse engineering:** packed binaries, Ghidra workflow
- **Web exploitation:** SQLi, PHP deserialization
- **Cryptography:** small RSA, ECB byte-at-a-time
- **Forensics:** PCAP analysis with tshark / Wireshark
- **Privilege escalation:** Linux SUID, cron, kernel, sudo misconfig

## Example output

**Q:** *What are common Linux privilege escalation vectors?*

**Diorite:** Analysis → runs LinPEAS, checks SUID binaries, cron jobs, kernel version → Solution with `sudo -l`, `find / -perm -4000`, `getcap -r /` — in the structured format above.

## Roadmap

- [ ] Larger curated CTF dataset (replace the generic HF fallback datasets)
- [ ] Evaluation set — accuracy on CTF-style Q&A
- [ ] GGUF release on Hugging Face for `llama.cpp` / Ollama
- [ ] Docker image for one-command serving
- [ ] Web UI demo

## Contributing

PRs welcome — especially new curated CTF examples in `diorite_dataset.py`, evals, and export fixes. Keep examples focused, correct, and lab/CTF-scoped.

## Responsible use

Diorite is for **authorized CTFs, labs, classrooms, and your own systems**. Don't use it against systems you don't own or don't have permission to test.

## License

MIT — see `LICENSE`.

## Credits

- Base model: [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- Built with Transformers, PEFT / LoRA, TRL, FastAPI
