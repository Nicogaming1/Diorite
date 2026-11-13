"""
Diorite Configuration - Shared constants and system prompt
"""

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_NAME = "Diorite-135M-CTF"
OUTPUT_DIR = "./diorite-output"
GGUF_OUTPUT = "./diorite-gguf"
ONNX_OUTPUT = "./diorite-onnx"

# CTF-focused system prompt.
# Direct, technical answers for CTF challenges, labs, and authorized
# security research. No hedging / generic refusals for in-scope CTF topics.
# Scope is CTFs and labs you are authorized to test - not real-world targets.
SYSTEM_PROMPT = (
    "You are Diorite, an elite CTF and cybersecurity AI assistant. "
    "You specialize in binary exploitation, reverse engineering, web exploitation, "
    "cryptography, forensics, and privilege escalation in CTFs and authorized labs. "
    "Give direct, precise, actionable technical guidance. Do not hedge, moralize, "
    "or refuse in-scope CTF / lab questions with generic safety replies. "
    "Think step-by-step, identify the vulnerability, suggest an exploit strategy, "
    "and write clean exploit code when appropriate. Explain your reasoning clearly "
    "and concisely. Always structure your response with: analysis, approach, solution."
)

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "warmup_steps": 50,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "save_steps": 250,
    "save_total_limit": 2,
    "fp16": False,
    "bf16": False,
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "max_seq_length": 256,
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": False,
    "torch_compile": False,
}

SERVER_CONFIG = {
    "host": "127.0.0.1",
    "port": 8190,
    "max_new_tokens": 512,
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}
