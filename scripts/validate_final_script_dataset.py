"""Validate generated script-classifier CSVs before fine-tuning.

Checks label/script consistency, schema, empty values, normalized duplicate overlap,
split-group leakage, metadata-to-CSV consistency, class balance, and source mix.
Run from the repository root after scripts/build_final_script_dataset.py.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

OUT = Path("data/processed/script_classifier")
META = Path("data/interim/script_classifier/deduplicated_candidates_with_metadata.csv")
REPORT_PATH = OUT / "validation_report.json"
PARTS = ("train", "validation", "test")
DEV_RE = re.compile(r"[\u0900-\u097F]")
LAT_RE = re.compile(r"[A-Za-z]")
WS_RE = re.compile(r"\s+")


def clean(value: object) -> str:
    return WS_RE.sub(" ", unicodedata.normalize("NFC", str(value)).replace("\ufeff", "")).strip()


def norm_key(text: object) -> str:
    value = clean(text).casefold()
    value = re.sub(r"[^\w\u0900-\u097f]+", " ", value, flags=re.UNICODE)
    return WS_RE.sub(" ", value).strip()


def expected_label(text: str) -> str | None:
    has_dev = bool(DEV_RE.search(text))
    has_lat = bool(LAT_RE.search(text))
    if has_dev:
        return "DEVANAGARI"
    if has_lat:
        return "LATIN"
    return None


def main() -> int:
    errors: list[str] = []
    report: dict = {"partitions": {}, "checks": {}, "errors": errors}
    frames: dict[str, pd.DataFrame] = {}

    for part in PARTS:
        path = OUT / f"{part}.csv"
        if not path.exists():
            errors.append(f"Missing partition file: {path}")
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        frames[part] = df
        if list(df.columns) != ["text", "script"]:
            errors.append(f"{part}: expected exactly columns ['text', 'script']; got {list(df.columns)}")
            continue
        if (df["text"].str.strip() == "").any():
            errors.append(f"{part}: contains empty text")
        invalid_labels = sorted(set(df["script"]) - {"LATIN", "DEVANAGARI"})
        if invalid_labels:
            errors.append(f"{part}: invalid labels {invalid_labels}")

        mismatches = []
        for idx, row in df.iterrows():
            expected = expected_label(row["text"])
            if expected is None or row["script"] != expected:
                mismatches.append({
                    "row": int(idx) + 2,
                    "script": row["script"],
                    "expected": expected,
                    "text": row["text"][:180],
                })
        if mismatches:
            errors.append(f"{part}: {len(mismatches)} label/script mismatches or unsupported-script rows")
        keys = df["text"].map(norm_key)
        dup_count = int(keys.duplicated(keep=False).sum())
        if dup_count:
            errors.append(f"{part}: {dup_count} rows participate in normalized duplicate collisions")
        report["partitions"][part] = {
            "rows": int(len(df)),
            "labels": {str(k): int(v) for k, v in df["script"].value_counts().items()},
            "label_script_mismatches": len(mismatches),
            "normalized_duplicate_rows": dup_count,
            "columns": list(df.columns),
        }

    if len(frames) == 3:
        key_sets = {p: set(df["text"].map(norm_key)) for p, df in frames.items()}
        overlap = {}
        for i, left in enumerate(PARTS):
            for right in PARTS[i + 1:]:
                shared = key_sets[left] & key_sets[right]
                overlap[f"{left}__{right}"] = len(shared)
                if shared:
                    errors.append(f"Normalized text leakage between {left} and {right}: {len(shared)} keys")
        report["checks"]["normalized_text_overlap"] = overlap

    if not META.exists():
        errors.append(f"Missing provenance metadata: {META}")
    else:
        meta = pd.read_csv(META, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        required = {"text", "script", "partition", "split_group", "source_dataset"}
        missing = sorted(required - set(meta.columns))
        if missing:
            errors.append(f"Metadata missing required columns: {missing}")
        else:
            # Group IDs must never be shared by partitions.
            group_parts = meta.groupby("split_group")["partition"].nunique()
            leaked_groups = int((group_parts > 1).sum())
            if leaked_groups:
                errors.append(f"{leaked_groups} split_group IDs cross partitions")
            report["checks"]["split_group_leakage_groups"] = leaked_groups

            # Every final CSV must match the corresponding metadata partition.
            for part, df in frames.items():
                if not {"text", "script"}.issubset(df.columns):
                    continue
                csv_pairs = set(zip(df["text"].map(norm_key), df["script"]))
                mpart = meta[meta["partition"] == part]
                meta_pairs = set(zip(mpart["text"].map(norm_key), mpart["script"]))
                if csv_pairs != meta_pairs:
                    errors.append(
                        f"{part}: CSV/metadata mismatch (CSV-only={len(csv_pairs-meta_pairs)}, "
                        f"metadata-only={len(meta_pairs-csv_pairs)})"
                    )

            source_mix: dict[str, dict[str, int]] = {}
            for part in PARTS:
                mpart = meta[meta["partition"] == part]
                counts: Counter[str] = Counter()
                for value in mpart["source_dataset"]:
                    counts.update(s for s in value.split(";") if s)
                source_mix[part] = dict(counts)
            report["checks"]["source_mix_by_partition"] = source_mix
            report["checks"]["manual_review_rows_by_partition"] = {
                part: int(((meta["partition"] == part) & (meta.get("needs_manual_review", "False").astype(str).str.casefold() == "true")).sum())
                if "needs_manual_review" in meta.columns else 0
                for part in PARTS
            }

    total = sum(len(df) for df in frames.values())
    report["checks"]["total_rows"] = total
    report["checks"]["split_proportions"] = {
        part: round(len(frames[part]) / total, 4) if total else 0
        for part in PARTS if part in frames
    }
    report["passed"] = not errors
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nValidation report: {REPORT_PATH}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
