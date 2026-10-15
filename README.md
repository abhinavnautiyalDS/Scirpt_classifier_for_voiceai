# Hinglish Language Switch LLM

A small language-identification model and policy layer for voice agents that need to decide whether a caller is speaking **Hinglish/Hindi**, **English**, or whether there is not enough evidence to switch.

> **Status:** experimental LoRA adapter trained. A provisional test run reported 96.58% accuracy and 0.9427 macro-F1 on 322 automatically labeled samples. These are agreement metrics against heuristic labels, not human-reviewed real-world accuracy. The adapter must be placed under `models/qwen-hinglish-lora/` for the Streamlit app to load it.

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
- `src/extract_callers.py` — extract caller turns only from timestamped plain-text call transcripts.
- `src/prepare_data.py` — validate labeled CSVs and create reproducible splits.
- `src/policy.py` — neutral phrase override and consecutive-confirmation switch controller.
- `src/train_lora.py` — Qwen2.5-0.5B-Instruct LoRA fine-tuning starter.
- `src/predict.py` — inference wrapper for a fine-tuned adapter.
- `src/evaluate.py` — held-out metrics and confusion matrix.
- `app.py` — Streamlit playground for manual utterance checks. It expects the extracted adapter in `models/qwen-hinglish-lora/` or `HINGLISH_MODEL_PATH` to point to an adapter directory.
- `tests/test_policy.py` and `tests/test_extract_callers.py` — basic policy and parser tests.
- `docs/voice_pipeline.md` — integration flow and switching safeguards.
- `data/example_utterances.csv` — small synthetic examples for schema smoke tests only; not a training benchmark.
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

First, extract caller turns from your local transcript file (do not commit real transcripts or extracted caller text to this public repository):

```bash
python src/extract_callers.py --input /path/to/call_transcript.txt --output /private/path/caller_turns.csv
```

Add and human-review labels in a private CSV at `data/utterances.csv` using the schema in `data/README.md`. Mark `reviewed=true` only after checking each label. Then prepare your **human-reviewed** splits:

```bash
python src/prepare_data.py --input data/utterances.csv --output-dir data/splits --require-reviewed
```

Run basic tests:

```bash
pytest -q
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

### Run the fine-tuned adapter in Streamlit

Extract the downloaded model ZIP and copy only the **top-level adapter files** into `models/qwen-hinglish-lora/`:

- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja` (if present)
- `README.md` (optional model notes)

Do not copy `checkpoint-198/`, `checkpoint-297/`, optimizer states, RNG states, or training checkpoints into the app directory. They are not needed for inference and make the repository unnecessarily large. The repository's `.gitignore` excludes model weights by default, so use `git add -f models/qwen-hinglish-lora/adapter_model.safetensors` when deliberately adding the trained adapter.

On Streamlit Community Cloud, select `app.py` as the entry point. The app downloads the base model `Qwen/Qwen2.5-0.5B-Instruct` when it first loads; cold start can take time and the available RAM may be a limitation. Keep the repository and model weights private if the data or model is not intended for public redistribution.

Training can be GPU-intensive. Fine-tuning is intended for a GPU; CPU inference should be benchmarked on the target machine. Do not treat validation metrics on automatically labeled data as a reliable accuracy estimate.

## Evaluation checklist

1. Split by call or speaker/session group when possible, not random utterance alone.
2. Human-review labels, especially code-mixed, short, noisy, and unsupported-language examples.
3. Keep a locked test set that is not used for prompt/model selection.
4. Report macro-F1, per-class precision/recall/F1, confusion matrix, neutral false-switch rate, and latency on the target CPU.
5. Use `LanguageSwitchController` in `src/policy.py` to keep neutral/uncertain turns from switching and require repeated candidate labels. Tune its thresholds with held-out call-level tests.
6. Only claim the >96% target if a representative human-reviewed held-out test set supports it.

## License

No license has been selected yet. Add one before accepting external contributions or redistributing this project.
