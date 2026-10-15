#!/usr/bin/env python3
"""Evaluate exact-label predictions on a human-reviewed held-out CSV."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from .predict import LanguageSwitchClassifier, LABELS
except ImportError:
    from predict import LanguageSwitchClassifier, LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", required=True, help="LoRA adapter directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="artifacts/evaluation.json")
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.test).fillna({"text": "", "label": ""})
    if not {"text", "label"}.issubset(df.columns):
        raise SystemExit("Test CSV must contain text,label")
    if args.require_reviewed:
        if "reviewed" not in df.columns:
            raise SystemExit("Test CSV has no reviewed column")
        reviewed = df["reviewed"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
        if not reviewed.all():
            raise SystemExit("Test set includes rows not marked human-reviewed")
    df["label"] = df["label"].astype(str).str.upper()
    df = df[df["label"].isin(LABELS) & df["text"].astype(str).str.strip().ne("")]
    if df.empty:
        raise SystemExit("No usable test rows")

    classifier = LanguageSwitchClassifier(args.model, args.base_model)
    predictions, latencies = [], []
    for text in df["text"].astype(str):
        start = time.perf_counter()
        result = classifier.classify(text)
        latencies.append((time.perf_counter() - start) * 1000)
        predictions.append(result.label)
    report = classification_report(
        df["label"], predictions, labels=list(LABELS), output_dict=True, zero_division=0
    )
    sorted_latency = sorted(latencies)
    result = {
        "n": len(df),
        "accuracy": float(accuracy_score(df["label"], predictions)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(df["label"], predictions, labels=list(LABELS)).tolist(),
        "latency_ms_median": sorted_latency[len(sorted_latency) // 2],
        "latency_ms_p95": sorted_latency[min(len(sorted_latency) - 1, int(len(sorted_latency) * 0.95))],
        "note": "Metrics are credible only for a representative human-reviewed test set held out before model/prompt iteration. Latency depends on hardware and includes generation.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n", "accuracy", "macro_f1", "latency_ms_median", "latency_ms_p95")}, indent=2))
    print(f"Full report written to {output}")


if __name__ == "__main__":
    main()
