# Baseline results — provisional only

A CPU-friendly TF-IDF + LinearSVC baseline was trained locally on the provisional train split to establish a comparison point while Qwen LoRA fine-tuning awaits a GPU runtime.

- Train / validation / test rows: 1,574 / 317 / 322
- Provisional-label test accuracy: 94.72%
- Provisional-label test macro-F1: 0.866
- Provisional-label test weighted-F1: 0.942
- CPU prediction time observed in this runtime: about 0.05 ms per utterance (not including application overhead)
- Human-reviewed rows: 0

**Do not report these as validated model performance.** All source labels are heuristic and the model is learning to reproduce those heuristics. The UNCERTAIN class recall is only about 0.41 on this split. Human-review the labels, lock a call-separated test set, and evaluate again before making any accuracy claim.

This is a classical baseline, not the requested fine-tuned LLM. The Qwen2.5-0.5B LoRA script and GPU notebook remain the intended LLM path. This runtime has CPU-only PyTorch and no cached Qwen weights, so it cannot perform that GPU fine-tuning here.
