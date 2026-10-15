#!/usr/bin/env python3
"""Train a CPU-friendly TF-IDF/LinearSVC baseline; this is not an LLM fine-tune."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

LABELS = ["HINGLISH", "ENGLISH", "NEUTRAL", "UNCERTAIN"]

def load(path):
    df = pd.read_csv(path).fillna({"text": "", "label": ""})
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"{path} must contain text,label columns")
    df["text"] = df.text.astype(str).str.strip()
    df["label"] = df.label.astype(str).str.upper().str.strip()
    return df[(df.text != "") & df.label.isin(LABELS)].copy()

def build_model():
    return Pipeline([
        ("features", FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=250000)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", min_df=1, sublinear_tf=True, max_features=100000)),
        ])),
        ("clf", LinearSVC(C=1.5, class_weight="balanced")),
    ])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output-dir", default="artifacts/baseline")
    p.add_argument("--require-reviewed", action="store_true")
    args = p.parse_args()
    paths = (args.train, args.validation, args.test)
    frames = [load(path) for path in paths]
    if args.require_reviewed:
        for path in paths:
            raw = pd.read_csv(path)
            if "reviewed" not in raw or not raw.reviewed.astype(str).str.lower().isin({"true", "1", "yes", "y"}).all():
                raise SystemExit(f"{path}: every row must be human-reviewed")
    train, validation, test = frames
    model = build_model()
    start = time.perf_counter()
    model.fit(train.text, train.label)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(test.text)
    predict_seconds = time.perf_counter() - start
    val_pred = model.predict(validation.text)
    report = classification_report(test.label, pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics = {
        "model": "TF-IDF char 2-5 and word 1-2 n-grams + LinearSVC",
        "is_llm": False,
        "label_source": "Metrics are not trustworthy estimates of real-world accuracy unless every label is human-reviewed.",
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "test_accuracy": accuracy_score(test.label, pred),
        "test_macro_f1": f1_score(test.label, pred, average="macro", labels=LABELS, zero_division=0),
        "test_weighted_f1": f1_score(test.label, pred, average="weighted", labels=LABELS, zero_division=0),
        "test_classification_report": report,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix_true_rows_pred_cols": confusion_matrix(test.label, pred, labels=LABELS).tolist(),
        "validation_accuracy": accuracy_score(validation.label, val_pred),
        "validation_macro_f1": f1_score(validation.label, val_pred, average="macro", labels=LABELS, zero_division=0),
        "fit_seconds": fit_seconds,
        "test_prediction_seconds": predict_seconds,
        "test_prediction_ms_per_utterance": predict_seconds * 1000 / max(1, len(test)),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "baseline.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    result = test[["text", "label"]].copy()
    result["predicted_label"] = pred
    result["correct"] = result.label == result.predicted_label
    if "call_id" in test:
        result.insert(0, "call_id", test.call_id)
    result.to_csv(out / "test_predictions.csv", index=False)
    print(json.dumps({k: v for k, v in metrics.items() if k != "test_classification_report"}, indent=2))
    print("Saved artifacts to", out)

if __name__ == "__main__":
    main()
