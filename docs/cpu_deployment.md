# CPU deployment and latency

## Benchmark before choosing a runtime

The base model is Qwen2.5-0.5B-Instruct with a LoRA adapter. Fine-tuning should happen on a GPU; deployment should be measured on the actual target CPU and workload.

Measure separately:
- model load time (cold start),
- first utterance latency,
- steady-state p50/p95 per-utterance latency,
- memory usage,
- neutral false-switch rate,
- class-wise precision/recall and macro-F1.

The `src/evaluate.py` latency numbers use the Transformers/PyTorch runtime and include text generation. They are hardware-specific, not universal guarantees.

## Recommended CPU optimization experiment

1. Train the adapter and evaluate the unquantized model.
2. Merge the adapter into the base model in a controlled environment using PEFT's `merge_and_unload()`.
3. Export the merged model to GGUF with a compatible llama.cpp version.
4. Quantize to Q4_K_M (and optionally compare Q5_K_M) using llama.cpp's quantizer.
5. Run the same held-out utterance benchmark through llama.cpp and compare quality, p50/p95 latency, RAM, and neutral false-switch rate.
6. Keep the deterministic policy in `src/policy.py` after model inference.

Do not assume quantization preserves accuracy; re-run the full test suite after export. The model's output labels and prompt must remain consistent across runtimes.

## Operational guidance

- Keep the model resident in memory; do not reload it per utterance.
- Invoke classification after a final ASR endpoint, not on every partial token, unless streaming experiments prove it is safe.
- Use a stable current-language controller and retain the existing language for NEUTRAL/UNCERTAIN.
- Log decision metadata and timing; avoid storing raw caller transcripts unnecessarily.
