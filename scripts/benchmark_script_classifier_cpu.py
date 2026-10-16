"""Benchmark single-utterance CPU inference for a saved script classifier.

Example:
python scripts/benchmark_script_classifier_cpu.py --model-dir artifacts/script_classifier_bert_tiny/model --input-csv data/processed/script_classifier/test.csv --samples 1000
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-csv", default="data/processed/script_classifier/test.csv")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--threads", type=int, default=0, help="0 uses PyTorch default")
    parser.add_argument("--output", default="artifacts/script_classifier_bert_tiny/cpu_benchmark.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads if args.threads > 0 else torch.get_num_threads())
    torch.set_grad_enabled(False)
    model_dir = Path(args.model_dir)
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "text" not in df.columns or df.empty:
        raise ValueError(f"Input CSV must contain a non-empty text column: {args.input_csv}")
    texts = df["text"].tolist()
    if args.samples > 0:
        texts = texts[: min(args.samples, len(texts))]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to("cpu")
    model.eval()

    def infer(text):
        start = time.perf_counter()
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=args.max_length
        )
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs).item())
        elapsed_ms = (time.perf_counter() - start) * 1000
        return pred_id, probs.tolist(), elapsed_ms

    for text in texts[: min(args.warmup, len(texts))]:
        infer(text)

    latencies = []
    predictions = []
    run_start = time.perf_counter()
    for text in texts:
        pred, probs, latency = infer(text)
        predictions.append({"predicted_id": pred, "probabilities": probs})
        latencies.append(latency)
    elapsed = time.perf_counter() - run_start
    size_bytes = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())

    report = {
        "model_dir": str(model_dir),
        "device": "CPU",
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "measured_samples": len(texts),
        "warmup_samples": min(args.warmup, len(texts)),
        "max_length": args.max_length,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "median_p50": round(float(np.percentile(latencies, 50)), 3) if latencies else None,
            "p95": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
            "p99": round(float(np.percentile(latencies, 99)), 3) if latencies else None,
            "min": round(min(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "throughput_utterances_per_second": round(len(texts) / elapsed, 2) if elapsed else None,
        "model_directory_size_mb": round(size_bytes / (1024 * 1024), 3),
        "note": "Latency is end-to-end per utterance, including tokenization and CPU model forward pass. Use the same benchmark, input CSV, sample count, and thread count for baseline and fine-tuned models.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
