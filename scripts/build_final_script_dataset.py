"""Build a reviewed, two-label script-classification dataset from downloaded sources.

Input paths are the files produced by scripts/download_script_datasets.py.
Outputs: train.csv, validation.csv, test.csv (each exactly text,script), plus
an audit report and a provenance-bearing intermediate candidate file.

Policy:
- any retained text containing Devanagari is DEVANAGARI (including mixed script)
- Latin-only text is LATIN
- Trilingual/HinGE Hindi columns are Devanagari-source candidates; mixed rows are
  retained as DEVANAGARI, Latin-only rows are rejected
- English columns contribute only Latin-only examples; script-contaminated rows
  are rejected rather than silently mislabeled
- COMI-LINGUA contributes intact mixed-script sentences only, requiring >=2 of
  the 3 human annotators to mark both hi and en in token annotations
- Romanized Hinglish columns are never used
"""
from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path("data/raw")
OUT = Path("data/processed/script_classifier")
INTERIM = Path("data/interim/script_classifier")
DEV_RE = re.compile(r"[\u0900-\u097F]")
LAT_RE = re.compile(r"[A-Za-z]")
WS_RE = re.compile(r"\s+")


def clean(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return WS_RE.sub(" ", unicodedata.normalize("NFC", str(value)).replace("\ufeff", "")).strip()


def script_type(text):
    dev, lat = bool(DEV_RE.search(text)), bool(LAT_RE.search(text))
    if dev and lat:
        return "MIXED_SCRIPT"
    if dev:
        return "DEVANAGARI_ONLY"
    if lat:
        return "LATIN_ONLY"
    return "OTHER"


def parse_tags(value):
    """Parse a COMI annotation cell safely; return set of language tags or None."""
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    tags = set()
    for item in parsed:
        if isinstance(item, dict):
            tag = str(item.get("value", "")).strip().casefold()
            if tag in {"hi", "en"}:
                tags.add(tag)
    return tags


def read_csv(path):
    return pd.read_csv(path, dtype=object, keep_default_na=False, encoding="utf-8-sig")


def make_record(text, label, source, split, row_id, group_id, source_column, kind, review=False):
    return {
        "text": text,
        "script": label,
        "source_dataset": source,
        "source_split": split,
        "source_row_id": str(row_id),
        "group_id": str(group_id),
        "source_column": source_column,
        "script_type": kind,
        "needs_manual_review": bool(review),
    }


def extract_parallel_file(path, source, split, id_column, hindi_column, english_column, records, report):
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else read_csv(path)
    source_stats = {"raw_rows": len(df), "hindi_kept": 0, "hindi_rejected": Counter(),
                    "english_kept": 0, "english_rejected": Counter()}
    for idx, row in df.iterrows():
        rid = clean(row[id_column]) if id_column and id_column in df.columns else str(idx)
        # A Hindi-column sentence is valid for DEVANAGARI when Devanagari is present.
        text = clean(row[hindi_column])
        kind = script_type(text)
        if not text or kind == "OTHER":
            source_stats["hindi_rejected"][kind if text else "EMPTY"] += 1
        elif kind == "LATIN_ONLY":
            source_stats["hindi_rejected"][kind] += 1
        else:
            records.append(make_record(text, "DEVANAGARI", source, split, rid,
                                       f"{source}:{rid}", hindi_column, kind))
            source_stats["hindi_kept"] += 1

        # English-source examples are intentionally pure Latin only. Mixed/Devanagari
        # rows are source-label contamination, not reliable English training samples.
        text = clean(row[english_column])
        kind = script_type(text)
        if kind == "LATIN_ONLY":
            records.append(make_record(text, "LATIN", source, split, rid,
                                       f"{source}:{rid}", english_column, kind))
            source_stats["english_kept"] += 1
        else:
            source_stats["english_rejected"][kind if text else "EMPTY"] += 1

    report[f"{source}_{split}"] = {
        "raw_rows": source_stats["raw_rows"],
        "hindi_kept": source_stats["hindi_kept"],
        "hindi_rejected": dict(source_stats["hindi_rejected"]),
        "english_kept": source_stats["english_kept"],
        "english_rejected": dict(source_stats["english_rejected"]),
    }


def extract_comi(path, split, records, report):
    df = read_csv(path)
    sentence_col = next((c for c in df.columns if c.strip().casefold() == "sentences"), None)
    annotators = [c for c in df.columns if "annotated by: annotator" in c.casefold()]
    if not sentence_col or len(annotators) < 3:
        raise ValueError(f"Unexpected COMI schema: {list(df.columns)}")
    stats = Counter()
    for idx, row in df.iterrows():
        text = clean(row[sentence_col])
        kind = script_type(text)
        if not text:
            stats["empty"] += 1
            continue
        if kind != "MIXED_SCRIPT":
            stats[f"not_mixed_{kind}"] += 1
            continue

        valid_annotators = 0
        parseable_annotators = 0
        for col in annotators[:3]:
            tags = parse_tags(row[col])
            if tags is not None:
                parseable_annotators += 1
                if {"hi", "en"}.issubset(tags):
                    valid_annotators += 1

        if parseable_annotators < 2:
            stats["unparseable_or_missing_annotations"] += 1
            continue
        if valid_annotators < 2:
            stats["insufficient_human_agreement"] += 1
            continue

        # Keep intact sentence; never reconstruct by joining annotation tokens.
        records.append(make_record(
            text, "DEVANAGARI", "COMI-LINGUA", split, idx,
            f"COMI-LINGUA:{split}:{idx}", sentence_col, kind,
            review=(valid_annotators == 2 or parseable_annotators < 3)
        ))
        stats["kept_mixed_with_human_agreement"] += 1

    report[f"COMI-LINGUA_{split}"] = {"raw_rows": len(df), **dict(stats)}


def norm_key(text):
    # Case- and punctuation-insensitive key for duplicate detection, while retaining original text.
    value = clean(text).casefold()
    value = re.sub(r"[^\w\u0900-\u097f]+", " ", value, flags=re.UNICODE)
    return WS_RE.sub(" ", value).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    records, report = [], {
        "policy": {
            "label_rule": "Any Devanagari character => DEVANAGARI; Latin-only => LATIN",
            "trilingual_hinge": "Hindi columns retained only if Devanagari present; English columns retained only if Latin-only",
            "comi": "Intact mixed-script sentence; at least 2/3 human annotators must include both hi and en tags",
            "romanized_hinglish_columns": "excluded",
            "split": "random group-aware 80/10/10 across all retained candidates; normalized duplicates and aligned Trilingual row variants grouped; source train/test labels do not dictate final partitions",
        }
    }

    for split in ["train", "test"]:
        p = ROOT / "trilingual" / f"{split}.parquet"
        if p.exists():
            extract_parallel_file(p, "Trilingual", split, "id", "hindi", "english", records, report)

    p = ROOT / "hinge" / "HinGE.csv"
    if p.exists():
        extract_parallel_file(p, "HinGE", "unsplit", None, "Hindi", "English", records, report)

    for split in ["train", "test"]:
        p = ROOT / "comi_lingua" / f"LID_{split}.csv"
        if p.exists():
            extract_comi(p, split, records, report)

    if not records:
        raise RuntimeError(f"No source files found under {ROOT.resolve()}")

    cand = pd.DataFrame(records)
    cand["norm_key"] = cand["text"].map(norm_key)
    cand = cand[cand["norm_key"] != ""].copy()

    # Deduplicate exact normalized text. If identical text receives conflicting
    # labels, drop it from training and record it rather than guessing.
    dedupe_rows = []
    conflicts = []
    conflict_candidate_rows = 0
    collapsed_exact_duplicates = 0
    for key, group in cand.groupby("norm_key", sort=False):
        labels = set(group["script"])
        if len(labels) != 1:
            conflict_candidate_rows += len(group)
            conflicts.append({"norm_key": key, "texts": group["text"].head(3).tolist(),
                              "labels": sorted(labels), "sources": sorted(set(group["source_dataset"]))})
            continue
        collapsed_exact_duplicates += len(group) - 1
        row = group.iloc[0].to_dict()
        row["source_dataset"] = ";".join(sorted(set(group["source_dataset"])))
        row["source_splits"] = ";".join(sorted(set(group["source_split"])))
        row["provenance_count"] = int(len(group))
        row["needs_manual_review"] = bool(group["needs_manual_review"].any())
        # Ensure duplicate text variants are grouped together across sources.
        row["group_id"] = "text:" + key
        dedupe_rows.append(row)

    dedup = pd.DataFrame(dedupe_rows)
    if dedup.empty:
        raise RuntimeError("No conflict-free candidates remained after deduplication.")

    # Group all exact duplicates and aligned Trilingual language pairs by source row.
    # Because dedup above has one row per text, connect texts sharing an original
    # source row through a union-find over candidate provenance.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    text_keys = list(dedup["norm_key"])
    text_keys_set = set(text_keys)
    for key in text_keys:
        find("text:" + key)
    for _, row in cand.iterrows():
        tkey = norm_key(row["text"])
        if tkey not in text_keys_set:
            continue
        union("text:" + tkey, "source:" + str(row["group_id"]))
    dedup["split_group"] = dedup["norm_key"].map(lambda k: find("text:" + k))

    # Split all retained groups together so the final partitions target 80/10/10.
    # This avoids accidentally inflating test size by reserving every row that came
    # from a source dataset's test split. Exact duplicates and aligned Trilingual
    # language pairs are already connected through split_group.
    if dedup["split_group"].nunique() < 3:
        raise RuntimeError("At least three independent split groups are required.")

    test_splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    pool_idx, test_idx = next(test_splitter.split(dedup, groups=dedup["split_group"]))
    test_groups = set(dedup.iloc[test_idx]["split_group"])
    pool = dedup.iloc[pool_idx].copy()
    dedup["partition"] = "pool"
    dedup.loc[dedup["split_group"].isin(test_groups), "partition"] = "test"

    # Validation should be 10% of the full dataset: 10/90 = 1/9 of the remaining pool.
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=(1 / 9), random_state=43)
    train_idx, val_idx = next(val_splitter.split(pool, groups=pool["split_group"]))
    train_groups = set(pool.iloc[train_idx]["split_group"])
    val_groups = set(pool.iloc[val_idx]["split_group"])
    dedup.loc[dedup["split_group"].isin(train_groups), "partition"] = "train"
    dedup.loc[dedup["split_group"].isin(val_groups), "partition"] = "validation"

    # Final files have exactly the requested two columns.
    for part in ["train", "validation", "test"]:
        part_df = dedup[dedup["partition"] == part][["text", "script"]].copy()
        part_df = part_df.sample(frac=1, random_state=42).reset_index(drop=True)
        part_df.to_csv(OUT / f"{part}.csv", index=False, encoding="utf-8-sig")

    # Provenance and audits stay separate from final model input files.
    dedup.drop(columns=["norm_key"], errors="ignore").to_csv(
        INTERIM / "deduplicated_candidates_with_metadata.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(conflicts).to_json(INTERIM / "label_conflicts.json", orient="records",
                                    force_ascii=False, indent=2)
    report["deduplication"] = {
        "candidate_rows_before_dedup": int(len(cand)),
        "conflicting_text_groups_excluded": int(len(conflicts)),
        "candidate_rows_excluded_for_label_conflicts": int(conflict_candidate_rows),
        "rows_after_exact_dedup": int(len(dedup)),
        "duplicates_collapsed": int(collapsed_exact_duplicates),
    }
    report["final_partitions"] = {}
    for part in ["train", "validation", "test"]:
        df = pd.read_csv(OUT / f"{part}.csv")
        report["final_partitions"][part] = {
            "rows": int(len(df)),
            "labels": {str(k): int(v) for k, v in df["script"].value_counts().items()},
            "columns": list(df.columns),
        }
    report["manual_review_rows"] = int(dedup["needs_manual_review"].sum())
    (OUT / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nFinal files:", *(str(OUT / f"{p}.csv") for p in ["train", "validation", "test"]), sep="\n- ")


if __name__ == "__main__":
    main()
