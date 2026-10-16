"""Fine-tune BERT-Tiny for DEVANAGARI-vs-LATIN script classification.

Run from repository root. Final test set is evaluated only after training/model selection.
Outputs are written under artifacts/script_classifier_bert_tiny/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
LABEL2ID = {"DEVANAGARI": 0, "LATIN": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
DEV_RE = r"[\u0900-\u097F]"
LAT_RE = r"[A-Za-z]"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed/script_classifier")
    p.add_argument("--output-dir", default="artifacts/script_classifier_bert_tiny")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU training (not recommended)")
    return p.parse_args()


def read_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if list(df.columns) != ["text", "script"]:
        raise ValueError(f"{path}: expected exactly columns text,script; got {list(df.columns)}")
    if df.empty or (df["text"].str.strip() == "").any():
        raise ValueError(f"{path}: empty dataset or empty text rows")
    unexpected = sorted(set(df["script"]) - set(LABEL2ID))
    if unexpected:
        raise ValueError(f"{path}: unexpected labels {unexpected}")
    df["label"] = df["script"].map(LABEL2ID).astype(int)
    return df


def script_masks(texts: pd.Series):
    dev = texts.str.contains(DEV_RE, regex=True)
    lat = texts.str.contains(LAT_RE, regex=True)
    return dev, lat


def tokenizer_audit(tokenizer, train_df, max_length):
    """Measure tokenizer [UNK] exposure by script type before spending GPU time."""
    sample = train_df.sample(n=min(3000, len(train_df)), random_state=42)
    unknown_id = tokenizer.unk_token_id
    results = {}
    for name, mask in {
        "contains_devanagari": sample["text"].str.contains(DEV_RE, regex=True),
        "contains_latin": sample["text"].str.contains(LAT_RE, regex=True),
        "mixed_script": sample["text"].str.contains(DEV_RE, regex=True)
            & sample["text"].str.contains(LAT_RE, regex=True),
        "latin_only": ~sample["text"].str.contains(DEV_RE, regex=True)
            & sample["text"].str.contains(LAT_RE, regex=True),
    }.items():
        texts = sample.loc[mask, "text"].tolist()
        if not texts:
            results[name] = {"rows": 0, "unk_token_fraction": None}
            continue
        encoded = tokenizer(texts, truncation=True, max_length=max_length, add_special_tokens=True)
        ids = encoded["input_ids"]
        token_count = sum(len(x) for x in ids)
        unk_count = sum(sum(1 for token_id in x if token_id == unknown_id) for x in ids) if unknown_id is not None else 0
        results[name] = {
            "rows": len(texts),
            "unk_tokens": int(unk_count),
            "total_tokens": int(token_count),
            "unk_token_fraction": round(unk_count / token_count, 6) if token_count else 0.0,
        }
    return results


def tokenize_frame(df, tokenizer, max_length):
    return tokenizer(
        df["text"].tolist(),
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "devanagari_precision": float(precision[0]),
        "devanagari_recall": float(recall[0]),
        "devanagari_f1": float(f1[0]),
        "latin_precision": float(precision[1]),
        "latin_recall": float(recall[1]),
        "latin_f1": float(f1[1]),
    }


def evaluate_predictions(df, predictions, probabilities, prefix):
    y_true = df["label"].to_numpy()
    y_pred = np.asarray(predictions, dtype=int)
    report = {
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["DEVANAGARI", "LATIN"],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix_labels_order": ["DEVANAGARI", "LATIN"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    dev, lat = script_masks(df["text"])
    mixed_mask = dev & lat
    latin_only_mask = ~dev & lat
    pure_devanagari_mask = dev & ~lat

    def subset_recall(mask, target_label):
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) == 0:
            return {"rows": 0, "recall": None}
        return {
            "rows": int(len(idx)),
            "recall": float(np.mean(y_pred[idx] == target_label)),
        }

    report["script_slice_metrics"] = {
        "mixed_script_recall_as_DEVANAGARI": subset_recall(mixed_mask, LABEL2ID["DEVANAGARI"]),
        "latin_only_correct_as_LATIN": subset_recall(latin_only_mask, LABEL2ID["LATIN"]),
        "latin_only_false_DEVANAGARI_rate": (
            float(np.mean(y_pred[np.flatnonzero(latin_only_mask.to_numpy())] == LABEL2ID["DEVANAGARI"]))
            if latin_only_mask.any() else None
        ),
        "pure_devanagari_correct_as_DEVANAGARI": subset_recall(pure_devanagari_mask, LABEL2ID["DEVANAGARI"]),
    }
    pred_df = df[["text", "script"]].copy()
    pred_df["predicted_script"] = [ID2LABEL[int(x)] for x in y_pred]
    pred_df["correct"] = pred_df["script"] == pred_df["predicted_script"]
    pred_df["p_devanagari"] = probabilities[:, LABEL2ID["DEVANAGARI"]].astype(float)
    pred_df["p_latin"] = probabilities[:, LABEL2ID["LATIN"]].astype(float)
    pred_df["contains_devanagari"] = dev.to_numpy()
    pred_df["contains_latin"] = lat.to_numpy()
    pred_df.to_csv(Path(prefix).with_suffix(".csv"), index=False, encoding="utf-8-sig")
    return report


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_split(data_dir / "train.csv")
    val_df = read_split(data_dir / "validation.csv")
    test_df = read_split(data_dir / "test.csv")

    if args.cpu:
        use_cuda = False
    else:
        use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("WARNING: CUDA is not available; training on CPU will be slow.")
    print(f"Model: {args.model_name}")
    print(f"Rows: train={len(train_df):,}, validation={len(val_df):,}, test={len(test_df):,}")
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    audit = tokenizer_audit(tokenizer, train_df, args.max_length)
    (output_dir / "tokenizer_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Tokenizer [UNK] audit:")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    dev_unk = audit.get("contains_devanagari", {}).get("unk_token_fraction")
    if dev_unk is not None and dev_unk > 0.25:
        print("WARNING: Devanagari-containing samples have high [UNK] exposure. Inspect tokenizer_audit.json before trusting results.")

    train_enc = tokenize_frame(train_df, tokenizer, args.max_length)
    val_enc = tokenize_frame(val_df, tokenizer, args.max_length)
    test_enc = tokenize_frame(test_df, tokenizer, args.max_length)

    from datasets import Dataset

    def to_dataset(encodings, frame):
        ds = Dataset.from_dict(encodings)
        ds = ds.add_column("labels", frame["label"].tolist())
        return ds

    train_ds = to_dataset(train_enc, train_df)
    val_ds = to_dataset(val_enc, val_df)
    test_ds = to_dataset(test_enc, test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=100,
        report_to="none",
        fp16=bool(use_cuda and torch.cuda.is_available()),
        dataloader_num_workers=2 if use_cuda else 0,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    started = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - started
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))
    trainer.save_state()

    # The locked test set is evaluated only after best checkpoint selection via validation.
    test_output = trainer.predict(test_ds, metric_key_prefix="test")
    test_logits = np.asarray(test_output.predictions)
    test_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    test_pred = np.argmax(test_logits, axis=1)
    test_metrics = evaluate_predictions(
        test_df, test_pred, test_probs, str(output_dir / "test_predictions")
    )
    val_output = trainer.predict(val_ds, metric_key_prefix="validation")
    val_logits = np.asarray(val_output.predictions)
    val_probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy()
    val_metrics = evaluate_predictions(
        val_df, np.argmax(val_logits, axis=1), val_probs, str(output_dir / "validation_predictions")
    )

    summary = {
        "base_model": args.model_name,
        "label2id": LABEL2ID,
        "hyperparameters": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "device": "CUDA" if use_cuda else "CPU",
        "rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "training_seconds": round(train_seconds, 2),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "tokenizer_audit": audit,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "notes": [
            "Test metrics were computed after selecting the checkpoint using validation macro-F1.",
            "Mixed-script recall treats any sample containing both Devanagari and Latin characters as DEVANAGARI by policy.",
            "This report does not measure production CPU latency; benchmark the saved model on the target Windows CPU.",
        ],
    }
    (output_dir / "training_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    trainer.save_metrics("train", train_result.metrics)
    print("\nFINAL TEST METRICS")
    print(json.dumps({
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "script_slice_metrics": test_metrics["script_slice_metrics"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "training_seconds": round(train_seconds, 2),
        "model_dir": str(output_dir / "model"),
        "report": str(output_dir / "training_report.json"),
    }, ensure_ascii=False, indent=2))
    print("\nTraining complete. Keep test_predictions.csv locked for analysis only after this run.")


if __name__ == "__main__":
    main()
