# Hinglish Language Switch LLM

A small language-identification model and policy layer for voice agents that need to decide whether a caller is speaking **Hinglish/Hindi**, **English**, or whether there is not enough evidence to switch.

> **Status:** starter repository. No model has been trained or validated yet. Accuracy targets (including >96%) are goals, not claims. Heuristic or automatically generated labels are not ground truth.

## Intended labels

| Label | Meaning | Switch? |
|---|---|---|
| `HINGLISH` | Hindi in Devanagari or Latin script, or meaningful Hindi-English code-mixing | Potentially, after policy checks |
| `ENGLISH` | Meaningful English utterance | Potentially, after policy checks |
| `NEUTRAL` | Generic acknowledgements/greetings or utterances too short to provide language evidence | No |
| `UNCERTAIN` | Unsupported language, unclear transcription, or insufficient confidence | No |

The model identifies the utterance; a separate controller decides whether switching is safe. The controller should retain the current language for `NEUTRAL` and `UNCERTAIN`, and should use minimum-length/evidence, confidence, and stability checks.

## Repository layout

- `data/README.md` — dataset schema, annotation guide, and privacy guidance.
- `src/prepare_data.py` — validate labeled CSVs and create reproducible splits.
- `src/train_lora.py` — Qwen2.5-0.5B-Instruct LoRA fine-tuning starter.
- `src/predict.py` — inference wrapper for a fine-tuned adapter.
- `src/evaluate.py` — held-out metrics and confusion matrix.
- `app.py` — Streamlit playground for manual utterance checks.
- `tests/test_policy.py` — policy behavior checks.
- `requirements.txt` — project dependencies.

## Dataset format

CSV columns:

- `text`: caller utterance only (not the agent's words).
- `label`: one of `HINGLISH`, `ENGLISH`, `NEUTRAL`, `UNCERTAIN`.
- `call_id`: optional call/group identifier used to avoid leaking turns from the same call across splits.
- `reviewed`: optional boolean indicating a human-reviewed label.

Keep real caller data private and remove personal information before training. Do not commit raw call transcripts or unreviewed sensitive data to this public repository. The project does not include any source call transcripts.

## Quick start

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Prepare your **human-reviewed** CSV at `data/utterances.csv`:

```bash
python src/prepare_data.py --input data/utterances.csv --output-dir data/splits --require-reviewed
```

Fine-tune on a machine with a supported GPU:

```bash
python src/train_lora.py --train data/splits/train.csv --validation data/splits/validation.csv --output-dir artifacts/qwen-hinglish-lora
```

Evaluate the adapter on a held-out, human-reviewed test set:

```bash
python src/evaluate.py --test data/splits/test.csv --model artifacts/qwen-hinglish-lora
```

Launch the manual playground:

```bash
streamlit run app.py
```

Training can be GPU-intensive. Fine-tuning is intended for a GPU; CPU inference should be benchmarked on the target machine. Do not treat validation metrics on automatically labeled data as a reliable accuracy estimate.

## Evaluation checklist

1. Split by call or speaker/session group when possible, not random utterance alone.
2. Human-review labels, especially code-mixed, short, noisy, and unsupported-language examples.
3. Keep a locked test set that is not used for prompt/model selection.
4. Report macro-F1, per-class precision/recall/F1, confusion matrix, and latency on the target CPU.
5. Measure switch false positives separately for neutral phrases and single-word fragments.
6. Only claim the >96% target if a representative human-reviewed held-out test set supports it.

## License

No license has been selected yet. Add one before accepting external contributions or redistributing this project.
