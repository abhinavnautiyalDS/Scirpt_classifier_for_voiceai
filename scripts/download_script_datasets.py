"""Download the three selected sources into data/raw/ for local auditing.

This downloads source files only; it does not label, merge, or train.
Review dataset licenses before using the results commercially.
"""
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main():
    # Trilingual corpus: preserve official splits and export rows for inspection.
    out = RAW / "trilingual"
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("debajyotimaz/trilingual-hinglish-corpus")
    for split, table in ds.items():
        path = out / f"{split}.parquet"
        table.to_parquet(path, index=False)
        print(f"Trilingual {split}: {len(table):,} rows -> {path}")
        print("Columns:", table.column_names)
        print(table.select(range(min(3, len(table)))).to_pandas().to_string(index=False))

    # HinGE: download original CSV.
    out = RAW / "hinge"
    out.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="LingoIITGN/HinGE", repo_type="dataset", filename="HinGE.csv",
        local_dir=str(out)
    )
    df = pd.read_csv(path, dtype=object, keep_default_na=False, encoding="utf-8-sig")
    print(f"HinGE: {len(df):,} rows -> {path}")
    print("Columns:", list(df.columns))
    print(df.head(3).to_string(index=False))

    # COMI-LINGUA LID: use intact sentence records and tags; no token concatenation.
    out = RAW / "comi_lingua"
    out.mkdir(parents=True, exist_ok=True)
    for filename in ["LID_train.csv", "LID_test.csv"]:
        path = hf_hub_download(
            repo_id="LingoIITGN/COMI-LINGUA", repo_type="dataset",
            filename=filename, local_dir=str(out)
        )
        df = pd.read_csv(path, dtype=object, keep_default_na=False, encoding="utf-8-sig")
        print(f"COMI-LINGUA {filename}: {len(df):,} rows -> {path}")
        print("Columns:", list(df.columns))
        print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
