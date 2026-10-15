#!/usr/bin/env python3
"""Run a trained TF-IDF/LinearSVC baseline on one utterance."""
import argparse, json
from pathlib import Path
import joblib

p = argparse.ArgumentParser()
p.add_argument("--model", default="artifacts/baseline/baseline.joblib")
p.add_argument("--text", required=True)
args = p.parse_args()
model = joblib.load(Path(args.model))
print(json.dumps({
    "text": args.text,
    "label": str(model.predict([args.text])[0]),
    "model_type": "classical baseline; not an LLM",
}, ensure_ascii=False))
