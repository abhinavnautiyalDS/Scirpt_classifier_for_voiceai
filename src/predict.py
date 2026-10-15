#!/usr/bin/env python3
"""Inference helper. Classification and permission to switch are separate concerns."""
from __future__ import annotations

import argparse
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .policy import apply_switch_policy
except ImportError:
    from policy import apply_switch_policy

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTION = (
    "Classify the caller utterance for a voice-agent language switch. "
    "HINGLISH means Hindi in Devanagari or Latin script, including Hindi-English code-mixing. "
    "ENGLISH means meaningful English. NEUTRAL means a generic greeting/acknowledgement "
    "or too little language evidence to trigger a switch. UNCERTAIN means unsupported "
    "language or unclear/ambiguous content. Return exactly one label and no explanation."
)


class LanguageSwitchClassifier:
    def __init__(self, adapter_path: str, base_model: str = DEFAULT_BASE_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(base_model)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.to(self.device).eval()

    def classify(self, text: str, current_language: str | None = None) -> dict:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if not clean:
            policy = apply_switch_policy(clean, "UNCERTAIN")
            return {"label": policy.label, "switch_candidate": False, "reason": policy.reason}

        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": clean},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids, max_new_tokens=5, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip().upper()
        match = re.search(r"\b(HINGLISH|ENGLISH|NEUTRAL|UNCERTAIN)\b", answer)
        predicted = match.group(1) if match else "UNCERTAIN"
        policy = apply_switch_policy(clean, predicted)
        return {
            "label": policy.label,
            "switch_candidate": policy.switch_candidate,
            "reason": policy.reason,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Fine-tuned LoRA adapter directory")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    result = LanguageSwitchClassifier(args.model, args.base_model).classify(args.text)
    print(result)


if __name__ == "__main__":
    main()
