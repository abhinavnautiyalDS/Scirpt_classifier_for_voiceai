#!/usr/bin/env python3
"""Validate labeled utterance CSVs and create call-group-separated splits."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

LABELS = {"HINGLISH", "ENGLISH", "NEUTRAL", "UNCERTAIN"}


def split_data(df: pd.DataFrame, seed: int):
    """Make ~71/14/14 splits using stratified group folds when possible.

    Call IDs are kept entirely within one split. Stratification improves label
    balance but cannot create examples for a class that is absent from the source.
    """
    df = df.reset_index(drop=True)
    if "call_id" in df.columns:
        groups = df["call_id"].fillna("").astype(str)
        if (groups != "").any() and groups.nunique() >= 7:
            groups = groups.where(groups != "", pd.Series([f"row-{i}" for i in range(len(df))]))
            splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
            folds = []
            for _, fold_idx in splitter.split(df, df["label"], groups=groups):
                folds.append(fold_idx)
            test_idx, validation_idx = folds[0], folds[1]
            train_idx = [i for fold_num, idx in enumerate(folds) if fold_num not in {0, 1} for i in idx]
            return (
                df.iloc[train_idx].sample(frac=1, random_state=seed).reset_index(drop=True),
                df.iloc[validation_idx].reset_index(drop=True),
                df.iloc[test_idx].reset_index(drop=True),
            )
        if groups.nunique() >= 3:
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
        raise ValueError("Too few examples per class for stratified splitting. Add examples or manually define splits.") from exc
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
    group_sets = []
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part.to_csv(output / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows; labels={part['label'].value_counts().to_dict()}")
        group_sets.append(set(part["call_id"].astype(str)) if "call_id" in part else set())
    if all(group_sets):
        overlap = (group_sets[0] & group_sets[1]) | (group_sets[0] & group_sets[2]) | (group_sets[1] & group_sets[2])
        print(f"Cross-split call_id overlap: {len(overlap)}")
        if overlap:
            raise SystemExit("Call-group leakage detected; refusing to finish.")
    print("Warning: a split does not become a valid benchmark just because it is balanced; labels must be human-reviewed.")


if __name__ == "__main__":
    main()
