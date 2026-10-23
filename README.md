# Hinglish Language Switch LLM

This repository contains two related experiments: a fine-tuned **BERT-Tiny script classifier** for `DEVANAGARI` vs `LATIN`, and an experimental Qwen LoRA language-switch classifier/policy prototype.

> **Current Streamlit app:** `app.py` loads the fine-tuned BERT-Tiny model from `artifacts/script_classifier_bert_tiny/model/` and displays its prediction, class probabilities, and inference latency. It does not currently run the four-class Qwen adapter.

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
- `app.py` — Streamlit playground for the fine-tuned BERT-Tiny script classifier; predicts `DEVANAGARI` or `LATIN`.
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

### Run the BERT-Tiny script classifier in Streamlit

The fine-tuned model files are committed under `artifacts/script_classifier_bert_tiny/model/`. Install dependencies from `requirements.txt`, then run:

```bash
streamlit run app.py
```

On Streamlit Community Cloud, choose this repository and `app.py` as the entry point. The app loads the local model files and shows the predicted script, both class probabilities, and inference latency.

The separate Qwen LoRA prototype remains in `src/` and `models/`, but is not the model used by the current Streamlit app.

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

---

## Branch focus: problem → change → evidence

**Branch-specific purpose:** Script dataset: prevent shortcut learning where Latin means English or Devanagari means Hindi; audit synthetic label quality and split leakage.


## Problem → solution → evidence: project evolution

### Product problem
A real-time voice agent uses STT transcripts to decide the language for the next response. Script alone is insufficient: Hindi can be Romanized, English and Hinglish can share Latin script, and a short “hello”, “okay” or “haan ji” may not justify switching.

### Experiments
- **Script classifier:** BERT-Tiny predicts DEVANAGARI vs LATIN. Useful as a script signal, but not a language detector.
- **Classical English/Hinglish:** TF-IDF character n-grams + LinearSVC with OTHER/UNCERTAIN rejection. Saved benchmark: 9,389 test rows, 97.2628% accuracy, 97.0005% macro-F1, p50 ≈ 1.337 ms, p95 ≈ 1.509 ms. Binary benchmark only; not comparable directly to 12-label results.
- **Qwen + LoRA:** explored context-aware classification. LoRA reduces trainable parameters, but inference still needs the base Qwen checkpoint. Exact parameter count, checkpoint and LoRA rank/alpha must be read from experiment config—not guessed.
- **BERT-Tiny E2/E4:** compact fixed-label model for a CPU-constrained use case. Current E2 v4.4 architecture: 2 layers, hidden size 128, 2 attention heads, intermediate size 512, vocab 30,522, 12 labels.
- **Version 4 data audit:** removed 9,992/10,000 flagged synthetic Punjabi rows reported to be mostly Hindi-looking Devanagari with Punjabi labels; trimmed repetitive templates, added STT-like noise and rebuilt group-aware splits.
- **Versions 4.1–4.3:** treat each archive as a separate snapshot. If the branch lacks a build report, state only verified counts and verify detailed historical changes against commits before claiming them as fact.
- **Version 4.4:** added 15,783 acknowledgement rows to none and relabeled 294 existing rows to none. Includes native/Romanized forms and case/punctuation variants. “Okay” alone is neutral; “Okay, I want to cancel my order” contains language-bearing evidence.

### Current v4.4 result (multilingual main project)
903,414 rows; 12 labels; 141 categories; train 723,240 / validation 90,066 / test 90,108. Uploaded BERT-Tiny E2 report: 99.868% accuracy and 99.868% macro-F1 on the prepared test split. The none class has lowest F1 (~99.446%). This is not proof of the same performance on live STT calls.

### Clickable references
- [COMI-LINGUA](https://github.com/lingo-iitgn/COMI-LINGUA) · [PHINC](https://huggingface.co/datasets/LingoIITGN/PHINC) · [Tatoeba English–Hindi](https://www.manythings.org/anki/hin-eng.zip)
- [AI4Bharat IndicLID](https://github.com/AI4Bharat/IndicLID)
- [Google BERT-Tiny checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) · [Qwen models](https://huggingface.co/Qwen)
- [Sarvam text language detection API](https://docs.sarvam.ai/api-reference/text/identify-language) · [fastText LID](https://fasttext.cc/docs/en/language-identification.html) · [CLD3](https://github.com/google/cld3) · [Lingua](https://github.com/pemistahl/lingua)

### Open-source release gates
Publish model quality (per-class metrics, confusion matrix, calibration and failures); dataset integrity (leakage, near duplicates, annotation and unseen data); CPU efficiency (p50/p90/p95, throughput, RAM, cold/warm start, concurrency); robustness (noise, typos, short/long text, mixed scripts, unsupported languages, malformed Unicode); fair same-sample benchmarks; API/timeouts/concurrency/rollback tests; dataset/model/dependency licensing, attribution and privacy checks; and a model card with reproducible evaluation, pinned install and known limitations. Separate hosted Sarvam network latency from local CPU inference.

