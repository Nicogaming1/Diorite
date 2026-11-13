"""
Diorite Trainer - LoRA Fine-tuning Script
Trains SmolLM2-135M-Instruct on CTF/cybersecurity data with hardcoded system prompt.
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from diorite_config import (
    BASE_MODEL,
    LORA_CONFIG,
    MODEL_NAME,
    OUTPUT_DIR,
    SYSTEM_PROMPT,
    TRAINING_CONFIG,
)
from diorite_dataset import build_training_dataset


def setup_tokenizer(model_name: str):
    """Load and configure tokenizer with proper padding."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int):
    """Tokenize dataset with truncation and padding."""

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1,
        desc="Tokenizing",
    )


def load_base_model(model_name: str):
    """Load the base causal LM model. fp16 on GPU, fp32 on CPU."""
    print(f"[+] Loading base model: {model_name}")
    use_fp16 = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def apply_lora(model, lora_config_dict: dict):
    """Apply LoRA adapters to the model."""
    print("[+] Applying LoRA configuration")
    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        lora_dropout=lora_config_dict["lora_dropout"],
        target_modules=lora_config_dict["target_modules"],
        bias=lora_config_dict["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(model, tokenized_dataset, tokenizer):
    """Run the fine-tuning loop."""
    tc = TRAINING_CONFIG

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        warmup_steps=tc["warmup_steps"],
        max_grad_norm=tc["max_grad_norm"],
        weight_decay=tc["weight_decay"],
        logging_steps=tc["logging_steps"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],
        fp16=tc["fp16"],
        bf16=tc["bf16"],
        optim=tc["optim"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=tc.get("dataloader_pin_memory", False),
        dataloader_num_workers=tc["dataloader_num_workers"],
        gradient_checkpointing=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("[+] Starting training")
    trainer.train()
    return trainer


def save_model(trainer, tokenizer):
    """Save the fine-tuned LoRA model and tokenizer."""
    output_path = Path(OUTPUT_DIR) / MODEL_NAME
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[+] Saving model to {output_path}")
    trainer.model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    meta_path = output_path / "diorite_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "base_model": BASE_MODEL,
                "system_prompt": SYSTEM_PROMPT,
                "lora_config": LORA_CONFIG,
                "training_config": TRAINING_CONFIG,
            },
            f,
            indent=2,
        )

    print("[+] Model saved. Files:")
    for p in output_path.iterdir():
        print(f"    {p.name} ({p.stat().st_size / 1024:.1f} KB)")


def test_inference(model, tokenizer):
    """Quick smoke test after training."""
    print("\n[+] Running inference test...")
    test_prompt = f"{SYSTEM_PROMPT}\n\n[INST] What is a buffer overflow?[/INST]"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[TEST OUTPUT]\n{response}\n")
    return response


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  DIORITE TRAINER - CTF/Cybersecurity LLM Fine-tuning")
    print(f"  Base Model: {BASE_MODEL}")
    print(f"  Output: {OUTPUT_DIR}/{MODEL_NAME}")
    print("=" * 60)

    tokenizer = setup_tokenizer(BASE_MODEL)

    print("\n[+] Building training dataset...")
    default_max = 200 if not torch.cuda.is_available() else 2000
    max_ex = int(os.environ.get("DIORITE_MAX_EXAMPLES", default_max))
    print(f"[+] Using up to {max_ex} training examples ({'GPU' if torch.cuda.is_available() else 'CPU'} mode)")
    raw_dataset = build_training_dataset(max_examples=max_ex)

    # FIX: use TRAINING_CONFIG max_seq_length (256), not a hardcoded 512.
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, max_length=TRAINING_CONFIG["max_seq_length"])

    model = load_base_model(BASE_MODEL)
    model = apply_lora(model, LORA_CONFIG)
    trainer = train(model, tokenized_dataset, tokenizer)
    save_model(trainer, tokenizer)
    test_inference(model, tokenizer)

    print("\n[+] Training complete!")
    print(f"[+] LoRA adapter saved to: {OUTPUT_DIR}/{MODEL_NAME}/")
    print("[+] Run diorite_converter.py to convert for deployment")


if __name__ == "__main__":
    main()
