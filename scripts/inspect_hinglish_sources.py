"""Audit/extract Hindi, English and mixed-script Hinglish source tables.

This is a data-preparation utility only; it does not train a model.
Download source files from their official dataset pages before running it.
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


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return WS_RE.sub(" ", unicodedata.normalize("NFC", str(value)).replace("\ufeff", "")).strip()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(obj if isinstance(obj, list) else obj.get("data", obj))
    return pd.read_csv(path, sep=None, engine="python", dtype=object, keep_default_na=False)


def schema_audit(name: str, df: pd.DataFrame, sample_n: int = 5) -> dict:
    return {
        "source": name,
        "rows": int(len(df)),
        "columns": [
            {
                "name": str(c),
                "dtype": str(df[c].dtype),
                "nonempty": int((df[c].astype(str).str.strip() != "").sum()),
            }
            for c in df.columns
        ],
        "samples": [
            {str(k): clean(v)[:300] for k, v in row.items()}
            for row in df.head(sample_n).to_dict(orient="records")
        ],
    }


def only_script_rows(df, text_col, label, source, source_type, group_col=None):
    rows = []
    stats = {"empty": 0, "wrong_script_or_mixed": 0, "retained": 0}
    for idx, row in df.iterrows():
        text = clean(row[text_col])
        if not text:
            stats["empty"] += 1
            continue
        has_dev = bool(DEV_RE.search(text))
        has_latin = bool(LAT_RE.search(text))
        valid = (has_dev and not has_latin) if label == "DEVANAGARI" else (has_latin and not has_dev)
        if not valid:
            stats["wrong_script_or_mixed"] += 1
            continue
        rows.append({
            "text": text,
            "script": label,
            "source_dataset": source,
            "source_type": source_type,
            "source_row_id": clean(row[group_col]) if group_col and group_col in df.columns else str(idx),
            "original_group_id": clean(row[group_col]) if group_col and group_col in df.columns else "",
        })
        stats["retained"] += 1
    return rows, stats


def mixed_script_candidates(df, sentence_col, source, group_col=None):
    """Filter intact sentence records only; never joins unrelated rows or tokens."""
    rows = []
    stats = {"empty": 0, "dev_only": 0, "latin_only": 0, "neither": 0, "both_scripts_candidate": 0}
    for idx, row in df.iterrows():
        text = clean(row[sentence_col])
        if not text:
            stats["empty"] += 1
            continue
        has_dev = bool(DEV_RE.search(text))
        has_latin = bool(LAT_RE.search(text))
        if has_dev and has_latin:
            stats["both_scripts_candidate"] += 1
            rows.append({
                "text": text,
                "script": "DEVANAGARI",
                "source_dataset": source,
                "source_type": "Mixed-script Hinglish candidate - linguistic review required",
                "source_row_id": clean(row[group_col]) if group_col and group_col in df.columns else str(idx),
                "original_group_id": clean(row[group_col]) if group_col and group_col in df.columns else "",
                "needs_language_tag_review": True,
            })
        elif has_dev:
            stats["dev_only"] += 1
        elif has_latin:
            stats["latin_only"] += 1
        else:
            stats["neither"] += 1
    return rows, stats


def exact_deduplicate(df: pd.DataFrame):
    if df.empty:
        return df, {"before": 0, "after": 0, "removed": 0}
    work = df.copy()
    work["_norm"] = work["text"].map(lambda text: clean(text).casefold())
    rows = []
    provenance_cols = ["source_dataset", "source_type", "source_row_id", "original_group_id"]
    for _, group in work.groupby("_norm", sort=False, dropna=False):
        first = group.iloc[0].drop(labels=["_norm"]).to_dict()
        for col in provenance_cols:
            if col in group.columns:
                values = []
                for value in group[col].astype(str):
                    if value and value not in values:
                        values.append(value)
                first[col] = ";".join(values)
        if "needs_language_tag_review" in group.columns:
            first["needs_language_tag_review"] = bool(group["needs_language_tag_review"].fillna(False).any())
        rows.append(first)
    result = pd.DataFrame(rows)
    return result, {"before": len(df), "after": len(result), "removed": len(df) - len(result)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trilingual", type=Path, help="Local Trilingual export (CSV/Parquet/JSON)")
    parser.add_argument("--hinge", type=Path, help="Local HinGE CSV/Parquet/JSON")
    parser.add_argument("--comi-lingua", type=Path, help="Local COMI-LINGUA task file")
    parser.add_argument("--comi-sentence-col", help="Exact sentence column after schema inspection")
    parser.add_argument("--comi-id-col", help="Optional original sentence/group ID column")
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim/source_audit"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged, audits = [], []
    report = {"sources": {}}

    if args.trilingual:
        df = read_table(args.trilingual)
        audits.append(schema_audit("Trilingual Hinglish Corpus", df))
        cols = {str(col).strip().casefold(): str(col) for col in df.columns}
        if not {"hindi", "english"}.issubset(cols):
            raise ValueError(f"Expected hindi/english columns; found {list(df.columns)}")
        id_col = cols.get("id")
        hindi, hindi_stats = only_script_rows(df, cols["hindi"], "DEVANAGARI",
                                              "trilingual-hinglish-corpus", "Hindi", id_col)
        english, english_stats = only_script_rows(df, cols["english"], "LATIN",
                                                  "trilingual-hinglish-corpus", "English", id_col)
        merged.extend(hindi + english)
        report["sources"]["trilingual"] = {
            "raw_rows": len(df), "hindi_filter": hindi_stats, "english_filter": english_stats,
            "excluded_columns": ["hinglish"],
        }

    if args.hinge:
        df = read_table(args.hinge)
        audits.append(schema_audit("HinGE", df))
        cols = {str(col).strip().casefold(): str(col) for col in df.columns}
        if not {"hindi", "english"}.issubset(cols):
            raise ValueError(f"Expected Hindi/English columns; found {list(df.columns)}")
        hindi, hindi_stats = only_script_rows(df, cols["hindi"], "DEVANAGARI", "HinGE", "Hindi")
        english, english_stats = only_script_rows(df, cols["english"], "LATIN", "HinGE", "English")
        merged.extend(hindi + english)
        report["sources"]["HinGE"] = {
            "raw_rows": len(df), "hindi_filter": hindi_stats, "english_filter": english_stats,
            "excluded_columns": ["Human-generated Hinglish", "WAC", "PAC", "other Hinglish variants"],
        }

    if args.comi_lingua:
        df = read_table(args.comi_lingua)
        audits.append(schema_audit("COMI-LINGUA", df))
        if not args.comi_sentence_col:
            raise ValueError("Inspect schema_report.json, then pass --comi-sentence-col with the actual intact sentence column.")
        if args.comi_sentence_col not in df.columns:
            raise ValueError(f"No column {args.comi_sentence_col!r}; found {list(df.columns)}")
        candidates, stats = mixed_script_candidates(
            df, args.comi_sentence_col, "COMI-LINGUA", args.comi_id_col
        )
        merged.extend(candidates)
        report["sources"]["COMI-LINGUA"] = {
            "raw_rows": len(df), "sentence_col": args.comi_sentence_col,
            "filter": stats,
            "warning": "Both-script candidates are not accepted as genuine Hinglish until language tags/context are reviewed.",
        }

    (args.out_dir / "schema_report.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["note"] = (
        "Counts apply only to the supplied local files. This script does not infer counts from dataset cards, "
        "does not reconstruct token-level COMI-LINGUA rows, does not split data, and does not train a model."
    )
    if merged:
        data = pd.DataFrame(merged)
        data, dedup_stats = exact_deduplicate(data)
        data.to_csv(args.out_dir / "merged_candidates_with_metadata.csv", index=False, encoding="utf-8-sig")
        # Final two-column view is created only for candidates that have passed the required manual/tag review.
        data[["text", "script"]].to_csv(
            args.out_dir / "two_column_preview_NOT_TRAINING_READY.csv", index=False, encoding="utf-8-sig"
        )
        report["exact_dedup"] = dedup_stats
        report["candidate_count"] = len(data)
        report["candidate_label_counts"] = {str(k): int(v) for k, v in data["script"].value_counts().items()}
        report["note"] += " The two-column file is a preview only; COMI-LINGUA candidates need language-tag/context validation first."
    (args.out_dir / "counts_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"schema_report": str(args.out_dir / "schema_report.json"),
                      "counts_report": str(args.out_dir / "counts_report.json"), **report},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
