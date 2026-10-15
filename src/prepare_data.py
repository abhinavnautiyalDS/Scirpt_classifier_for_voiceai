#!/usr/bin/env python3
"""Validate labeled utterance CSVs and create reproducible splits."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

LABELS = {"HINGLISH", "ENGLISH", "NEUTRAL", "UNCERTAIN"}


def split_data(df: pd.DataFrame, seed: int):
    """Create 70/15/15 splits; group by call_id when available to reduce leakage."""
    df = df.reset_index(drop=True)
    if "call_id" in df.columns:
        groups = df["call_id"].fillna("").astype(str)
        if (groups != "").any() and groups.nunique() >= 3:
            groups = groups.where(groups != "", pd.Series([f"row-{i}" for i in range(len(df))]))
            first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
            train_idx, hold_idx = next(first.split(df, groups=groups))
            train, hold = df.iloc[train_idx].copy(), df.iloc[hold_idx].copy()
            hold_groups = groups.iloc[hold_idx]
            second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
            val_idx, test_idx = next(second.split(hold, groups=hold_groups))
            return train.reset_index(drop=True), hold.iloc[val_idx].reset_index(drop=True), hold.iloc[test_idx].reset_index(drop=True)
    try:
        train, hold = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["label"])
        validation, test = train_test_split(hold, test_size=0.50, random_state=seed, stratify=hold["label"])
    except ValueError as exc:
        raise ValueError("Too few examples in one or more classes for stratified splitting. Add examples or create grouped splits manually.") from exc
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with text,label and optional call_id,reviewed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input).fillna({"text": "", "label": ""})
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[(df["text"] != "") & (df["label"] != "")].copy()
    bad = sorted(set(df["label"]) - LABELS)
    if bad:
        raise SystemExit(f"Unsupported labels: {bad}. Allowed labels: {sorted(LABELS)}")
    if args.require_reviewed:
        if "reviewed" not in df.columns:
            raise SystemExit("--require-reviewed requested but CSV has no reviewed column")
        reviewed = df["reviewed"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
        if not reviewed.all():
            raise SystemExit(f"{int((~reviewed).sum())} rows are not marked reviewed")
    if len(df) < 20:
        raise SystemExit(f"Only {len(df)} usable rows; collect more data before splitting")

    train, validation, test = split_data(df, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part.to_csv(output / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows; labels={part['label'].value_counts().to_dict()}")
    print("Inspect class counts and confirm groups do not overlap before training.")


if __name__ == "__main__":
    main()
