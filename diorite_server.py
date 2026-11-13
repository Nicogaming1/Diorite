"""
Diorite Server - FastAPI inference server
Serves the Diorite CTF/Cybersecurity LLM as a local REST API.

FIX vs original: dtype is now device-aware.
- GPU: float16 (fast)
- CPU: float32 (float16 training/inference is not supported on CPU and
  the old code forced float16 everywhere, which could fail on CPU-only machines)
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from diorite_config import (
    BASE_MODEL,
    MODEL_NAME,
    OUTPUT_DIR,
    SERVER_CONFIG,
    SYSTEM_PROMPT,
)


def get_dtype():
    """float16 on GPU, float32 on CPU."""
    return torch.float16 if torch.cuda.is_available() else torch.float32


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt/question")
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    system_prompt: Optional[str] = Field(default=None, description="Override system prompt")
    stream: bool = Field(default=False)


class GenerateResponse(BaseModel):
    response: str
    tokens_generated: int
    latency_ms: float
    model: str
    tokens_per_second: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    uptime_seconds: float


model = None
tokenizer = None
start_time = time.time()


def load_model():
    """Load the Diorite model (LoRA adapter or merged)."""
    global model, tokenizer
    dtype = get_dtype()
    print(f"[+] Device dtype: {dtype} ({'GPU' if torch.cuda.is_available() else 'CPU'})")

    lora_path = Path(OUTPUT_DIR) / MODEL_NAME
    merged_path = Path(OUTPUT_DIR) / f"{MODEL_NAME}-merged"

    if lora_path.exists():
        print(f"[+] Loading LoRA adapter from: {lora_path}")
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, str(lora_path))
        tokenizer = AutoTokenizer.from_pretrained(str(lora_path))

    elif merged_path.exists():
        print(f"[+] Loading merged model from: {merged_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(merged_path), torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(merged_path))

    else:
        print(f"[!] No trained model found. Loading base model: {BASE_MODEL}")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("[+] Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    print("[+] Shutting down Diorite server")


app = FastAPI(
    title="Diorite CTF AI Server",
    description="Ultra-lightweight CTF/Cybersecurity LLM inference server",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_name=MODEL_NAME,
        uptime_seconds=round(time.time() - start_time, 2),
    )


def _format_prompt(request: GenerateRequest) -> str:
    sys_prompt = request.system_prompt or SYSTEM_PROMPT
    return f"{sys_prompt}\n\n[INST] {request.prompt}[/INST]"


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = tokenizer(_format_prompt(request), return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=SERVER_CONFIG["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start

    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    tps = len(generated_tokens) / elapsed if elapsed > 0 else 0

    return GenerateResponse(
        response=response_text.strip(),
        tokens_generated=len(generated_tokens),
        latency_ms=round(elapsed * 1000, 2),
        model=MODEL_NAME,
        tokens_per_second=round(tps, 2),
    )


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from fastapi.responses import StreamingResponse

    inputs = tokenizer(_format_prompt(request), return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=SERVER_CONFIG["repetition_penalty"],
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def event_generator():
        for text in streamer:
            yield f"data: {json.dumps({'token': text, 'done': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat")
async def chat(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = tokenizer(_format_prompt(request), return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=SERVER_CONFIG["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    tps = len(generated_tokens) / elapsed if elapsed > 0 else 0

    return {
        "response": response_text.strip(),
        "model": MODEL_NAME,
        "latency_ms": round(elapsed * 1000, 2),
        "tokens_per_second": round(tps, 2),
    }


@app.get("/")
async def root():
    return {
        "name": "Diorite CTF AI Server",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "POST /generate": "Generate response to a prompt",
            "POST /generate/stream": "Stream response tokens",
            "POST /chat": "Chat-style endpoint",
            "GET /health": "Health check",
        },
    }


def main():
    print("=" * 60)
    print("  DIORITE SERVER - CTF/Cybersecurity LLM API")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Base: {BASE_MODEL}")
    print(f"  Address: {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    print("=" * 60)
    uvicorn.run(app, host=SERVER_CONFIG["host"], port=SERVER_CONFIG["port"], log_level="info")


if __name__ == "__main__":
    main()
