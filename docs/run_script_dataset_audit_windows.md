# Run the script-classifier dataset audit (Windows)

Use Python 3.12 in a virtual environment. These steps download source datasets only; no model training is performed.

## 1. Install dependencies

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas pyarrow datasets huggingface_hub
```

## 2. Download sources and print actual schemas

```powershell
python scripts/download_script_datasets.py
```

This writes source files under data/raw/: Trilingual official splits exported to Parquet, the original HinGE.csv, and COMI-LINGUA LID_train.csv and LID_test.csv.

Read the printed column names and representative rows. COMI-LINGUA LID uses intact sentence records and language annotations; do not concatenate tokens from different records.

## 3. Extract auditable candidates

```powershell
python scripts/extract_script_dataset.py --trilingual data/raw/trilingual/train.parquet --hinge data/raw/hinge/HinGE.csv --comi-lid data/raw/comi_lingua/LID_train.csv
```

If the downloaded Trilingual splits have different names, use the exact filenames created by step 2. To include the official COMI-LINGUA test file for additional candidate discovery, run a separate extraction and keep it separate from training data; never use it for model selection.

Outputs are written to data/interim/source_audit/: schema_report.json, counts_report.json, and merged_candidates_with_metadata.csv.

COMI-LINGUA mixed-script records are marked for review. This script deliberately does not create train/validation/test splits or a training-ready two-column CSV. Verify candidate labels, deduplicate across all source files, resolve group relationships, and review licensing before training.

## Notes
- Trilingual's reported CC BY-NC-SA 4.0 license may restrict commercial use.
- COMI-LINGUA's repository and dataset-card license statements should be reconciled before commercial use.
- Do not commit downloaded raw datasets or private real-call transcripts to this repository.
