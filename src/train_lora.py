#!/usr/bin/env python3
"""LoRA fine-tuning scaffold for Qwen2.5-0.5B-Instruct.

Training loss is applied only to assistant completion tokens, not the prompt.
Use a GPU runtime. Automatically labeled data is not a reliable evaluation target.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

LABELS = {"HINGLISH", "ENGLISH", "NEUTRAL", "UNCERTAIN"}
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTION = (
    "Classify the caller utterance for a voice-agent language switch. "
    "HINGLISH means Hindi in Devanagari or Latin script, including Hindi-English code-mixing. "
    "ENGLISH means meaningful English. NEUTRAL means a generic greeting/acknowledgement "
    "or too little language evidence to trigger a switch. UNCERTAIN means unsupported "
    "language or unclear/ambiguous content. Return exactly one label and no explanation."
)


class LabelDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.rows = []
        df = pd.read_csv(path).fillna({"text": "", "label": ""})
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError(f"{path} must contain text,label columns")
        skipped = 0
        for row in df.itertuples(index=False):
            text, label = str(row.text).strip(), str(row.label).strip().upper()
            if not text or label not in LABELS:
                continue
            prompt = [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": text},
            ]
            full = prompt + [{"role": "assistant", "content": label}]
            prefix_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            full_ids = tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False)
            if len(full_ids) > max_length:
                skipped += 1
                continue
            target_labels = list(full_ids)
            target_labels[:len(prefix_ids)] = [-100] * len(prefix_ids)
            if all(token == -100 for token in target_labels):
                continue
            self.rows.append({"input_ids": full_ids, "labels": target_labels})
        if skipped:
            print(f"Skipped {skipped} rows longer than max_length={max_length}; review if this is substantial.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


@dataclass
class CompletionCollator:
    tokenizer: object

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, labels, masks = [], [], []
        for feature in features:
            n = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * n)
            labels.append(feature["labels"] + [-100] * n)
            masks.append([1] * len(feature["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default="artifacts/qwen-hinglish-lora")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. Fine-tuning is intended for a supported GPU runtime.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    ))
    train_ds = LabelDataset(args.train, tokenizer, args.max_length)
    val_ds = LabelDataset(args.validation, tokenizer, args.max_length)
    if not len(train_ds) or not len(val_ds):
        raise SystemExit("No usable training/validation rows after filtering.")
    output = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=CompletionCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"Saved adapter and tokenizer to {output}")
    print("Next: evaluate against an untouched, human-reviewed test set.")


if __name__ == "__main__":
    main()
