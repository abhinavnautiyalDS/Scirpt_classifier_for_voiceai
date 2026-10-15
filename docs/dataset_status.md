# Dataset status (initial transcript audit)

The uploaded text transcript was parsed locally; raw transcript text and caller utterance CSVs are **not committed** to this public repository.

- Caller utterances extracted: 2,213
- Detected call IDs: 176
- Provisional heuristic labels: HINGLISH 1,412; ENGLISH 477; NEUTRAL 218; UNCERTAIN 106
- Rows flagged for human review: 345
- Rows marked human-reviewed: 0
- Provisional group-separated split sizes: train 1,574; validation 317; test 322
- Cross-split call ID overlap: 0

**These are dataset preparation counts, not model metrics. No LLM has been trained yet.** Automatic labels can be wrong, particularly for Romanized Hindi, short fragments, proper nouns, non-Hindi Devanagari text, and noisy ASR. Review labels and keep the test set locked before reporting accuracy.

The private review pack is generated in the conversation workspace and is not published to this repository. Use `notebooks/colab_finetune.ipynb` after producing a human-reviewed CSV and selecting a GPU runtime.
