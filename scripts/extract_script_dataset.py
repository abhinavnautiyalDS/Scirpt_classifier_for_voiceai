"""Download selected public datasets and extract script-classification candidates.

COMI-LINGUA LID records are sentence-preserving: use the intact Sentences field
and language tags; never concatenate tokens across rows. This script creates
auditable candidates, not a training-ready split.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

DEV_RE = re.compile(r"[\u0900-\u097F]")
LAT_RE = re.compile(r"[A-Za-z]")
WS_RE = re.compile(r"\s+")
HI_TAG = re.compile(r"(?<![A-Za-z])hi(?![A-Za-z])", re.I)
EN_TAG = re.compile(r"(?<![A-Za-z])en(?![A-Za-z])", re.I)


def clean(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return WS_RE.sub(" ", unicodedata.normalize("NFC", str(value)).replace("\ufeff", "")).strip()


def read_table(path):
    if path.suffix.casefold() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, dtype=False)
    if path.suffix.casefold() == ".json":
        return pd.read_json(path, dtype=False)
    return pd.read_csv(path, dtype=object, keep_default_na=False, encoding="utf-8-sig")


def audit(df, source):
    return {
        "source": source,
        "rows": int(len(df)),
        "columns": [{"name": str(c), "dtype": str(df[c].dtype),
                     "nonempty": int((df[c].astype(str).str.strip() != "").sum())}
                    for c in df.columns],
        "samples": [{str(k): clean(v)[:400] for k, v in row.items()}
                    for row in df.head(5).to_dict(orient="records")],
    }


def script_flags(text):
    return bool(DEV_RE.search(text)), bool(LAT_RE.search(text))


def extract_parallel(df, text_col, label, source, id_col=None):
    out, stats = [], {"empty": 0, "script_mismatch_or_mixed": 0, "retained": 0}
    for idx, row in df.iterrows():
        text = clean(row[text_col])
        if not text:
            stats["empty"] += 1
            continue
        has_dev, has_lat = script_flags(text)
        valid = (has_dev and not has_lat) if label == "DEVANAGARI" else (has_lat and not has_dev)
        if not valid:
            stats["script_mismatch_or_mixed"] += 1
            continue
        source_id = clean(row[id_col]) if id_col and id_col in df.columns else str(idx)
        out.append({"text": text, "script": label, "source_dataset": source,
                    "source_type": "Hindi" if label == "DEVANAGARI" else "English",
                    "source_row_id": source_id, "original_group_id": source_id})
        stats["retained"] += 1
    return out, stats


def find_col(df, exact_names):
    folded = {str(c).strip().casefold(): str(c) for c in df.columns}
    for name in exact_names:
        if name.casefold() in folded:
            return folded[name.casefold()]
    return None


def extract_comi_lid(df):
    sentence_col = find_col(df, ["Sentences", "Sentence", "sentence", "text"])
    if not sentence_col:
        raise ValueError(f"Cannot identify sentence column from schema: {list(df.columns)}")
    tag_cols = [c for c in df.columns if "predicted tags" in str(c).casefold()]
    annotator_cols = [c for c in df.columns if "annotator" in str(c).casefold()]
    # Use human annotations when available; retain predicted tags as a fallback signal.
    chosen_tag_cols = annotator_cols if annotator_cols else tag_cols
    if not chosen_tag_cols:
        raise ValueError(f"Cannot identify LID tag columns from schema: {list(df.columns)}")
    id_col = find_col(df, ["id", "sentence_id", "sample_id", "uid"])
    out, stats = [], {
        "empty_sentence": 0, "missing_or_unparseable_tags": 0,
        "no_hindi_english_tag_pair": 0, "not_both_scripts": 0,
        "mixed_script_tagged_candidates": 0
    }
    review_examples = []
    for idx, row in df.iterrows():
        text = clean(row[sentence_col])
        if not text:
            stats["empty_sentence"] += 1
            continue
        has_dev, has_lat = script_flags(text)
        if not (has_dev and has_lat):
            stats["not_both_scripts"] += 1
            continue
        tag_texts = [clean(row[c]) for c in chosen_tag_cols if clean(row[c])]
        if not tag_texts:
            stats["missing_or_unparseable_tags"] += 1
            continue
        # A tag source must mention both Hindi and English. If annotators disagree,
        # preserve the candidate but flag disagreement for review.
        per_source = [(bool(HI_TAG.search(t)), bool(EN_TAG.search(t))) for t in tag_texts]
        if not any(hi and en for hi, en in per_source):
            stats["no_hindi_english_tag_pair"] += 1
            continue
        source_id = clean(row[id_col]) if id_col and id_col in df.columns else str(idx)
        unanimous = all(hi and en for hi, en in per_source)
        out.append({
            "text": text, "script": "DEVANAGARI", "source_dataset": "COMI-LINGUA",
            "source_type": "Mixed-script Hinglish candidate",
            "source_row_id": source_id, "original_group_id": source_id,
            "annotation_columns_used": ";".join(chosen_tag_cols),
            "language_tag_review": "all_selected_tag_sources_include_hi_and_en" if unanimous else "annotator_disagreement_review",
            "needs_manual_review": True
        })
        stats["mixed_script_tagged_candidates"] += 1
        if len(review_examples) < 10:
            review_examples.append({"text": text[:300], "tag_values": tag_texts[:3]})
    return out, stats, {
        "sentence_column": sentence_col, "tag_columns": chosen_tag_cols,
        "id_column": id_col, "examples_for_manual_review": review_examples
    }


def dedupe_with_provenance(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"before": 0, "after": 0, "removed": 0}
    df["_key"] = df["text"].map(lambda x: clean(x).casefold())
    result = []
    for _, group in df.groupby("_key", sort=False, dropna=False):
        item = group.iloc[0].drop(labels=["_key"]).to_dict()
        for col in ["source_dataset", "source_type", "source_row_id", "original_group_id"]:
            vals = []
            for value in group[col].fillna("").astype(str) if col in group else []:
                if value and value not in vals:
                    vals.append(value)
            if vals:
                item[col] = ";".join(vals)
        if "needs_manual_review" in group:
            item["needs_manual_review"] = bool(group["needs_manual_review"].fillna(False).any())
        result.append(item)
    out = pd.DataFrame(result)
    return out, {"before": len(df), "after": len(out), "removed": len(df) - len(out)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trilingual", type=Path, help="Local Trilingual CSV; use the original export")
    p.add_argument("--hinge", type=Path, help="Local HinGE.csv")
    p.add_argument("--comi-lid", type=Path, help="Local COMI-LINGUA LID_train.csv and/or LID_test.csv")
    p.add_argument("--out-dir", type=Path, default=Path("data/interim/source_audit"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, audits, report = [], [], {"sources": {}}

    if args.trilingual:
        df = read_table(args.trilingual)
        audits.append(audit(df, "Trilingual Hinglish Corpus"))
        hcol, ecol = find_col(df, ["hindi"]), find_col(df, ["english"])
        if not hcol or not ecol:
            raise ValueError(f"Trilingual expected hindi and english columns; found {list(df.columns)}")
        id_col = find_col(df, ["id"])
        hrows, hs = extract_parallel(df, hcol, "DEVANAGARI", "trilingual-hinglish-corpus", id_col)
        erows, es = extract_parallel(df, ecol, "LATIN", "trilingual-hinglish-corpus", id_col)
        rows.extend(hrows + erows)
        report["sources"]["trilingual"] = {"raw_rows": len(df), "hindi": hs, "english": es,
                                             "excluded": ["hinglish (Romanized)"]}

    if args.hinge:
        df = read_table(args.hinge)
        audits.append(audit(df, "HinGE"))
        hcol, ecol = find_col(df, ["Hindi"]), find_col(df, ["English"])
        if not hcol or not ecol:
            raise ValueError(f"HinGE expected Hindi and English columns; found {list(df.columns)}")
        hrows, hs = extract_parallel(df, hcol, "DEVANAGARI", "HinGE")
        erows, es = extract_parallel(df, ecol, "LATIN", "HinGE")
        rows.extend(hrows + erows)
        report["sources"]["HinGE"] = {"raw_rows": len(df), "hindi": hs, "english": es,
                                        "excluded": ["Human-generated Hinglish", "WAC", "PAC"]}

    if args.comi_lid:
        df = read_table(args.comi_lid)
        audits.append(audit(df, "COMI-LINGUA LID"))
        crows, stats, schema = extract_comi_lid(df)
        rows.extend(crows)
        report["sources"]["COMI-LINGUA LID"] = {"raw_rows": len(df), "filter": stats, "schema": schema}

    (args.out_dir / "schema_report.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    report["note"] = "Counts describe only supplied local files. COMI mixed-script rows remain review candidates until language tags and sentence quality are verified."
    if rows:
        df, dedupe = dedupe_with_provenance(rows)
        df.to_csv(args.out_dir / "merged_candidates_with_metadata.csv", index=False, encoding="utf-8-sig")
        report["exact_dedup"] = dedupe
        report["candidate_rows"] = int(len(df))
        report["candidate_label_counts"] = {str(k): int(v) for k, v in df["script"].value_counts().items()}
        report["comi_rows_requiring_review"] = int(df.get("needs_manual_review", pd.Series(dtype=bool)).fillna(False).sum())
        # Do not write a training CSV until candidate review and grouped split checks pass.
    (args.out_dir / "counts_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"schema_report": str(args.out_dir / "schema_report.json"),
                      "counts_report": str(args.out_dir / "counts_report.json"), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
