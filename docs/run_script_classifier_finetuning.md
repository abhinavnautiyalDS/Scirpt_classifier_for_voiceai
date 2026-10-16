# Fine-tune and benchmark the script classifier

This workflow fine-tunes \`google/bert_uncased_L-2-H-128-A-2\` for the two labels \`DEVANAGARI\` and \`LATIN\`. It uses validation macro-F1 to select the checkpoint, evaluates the test set after training, reports mixed-script recall and English false-DEVANAGARI rate, and audits tokenizer \`[UNK]\` exposure.

## 1. Run in Colab

Pull the latest branch and install the training dependencies:

\`\`\`python
%cd /content/hinglish-language-switch-llm
!git pull origin script-classifier-dataset
!pip install -q "transformers>=4.46,<5" datasets accelerate scikit-learn pandas
\`\`\`

Enable a GPU in Colab (Runtime → Change runtime type → T4 GPU or another available GPU), then verify it:

\`\`\`python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
\`\`\`

The regenerated dataset must already exist at \`data/processed/script_classifier/\` and have passed \`scripts/validate_final_script_dataset.py\`.

## 2. Train and evaluate

\`\`\`python
!python scripts/train_script_classifier.py \\
  --data-dir data/processed/script_classifier \\
  --output-dir artifacts/script_classifier_bert_tiny \\
  --epochs 3 \\
  --max-length 96 \\
  --train-batch-size 64 \\
  --eval-batch-size 128 \\
  --learning-rate 3e-5
\`\`\`

If CUDA memory runs out, rerun with \`--train-batch-size 32 --eval-batch-size 64\`. The script uses validation macro-F1 for best-checkpoint selection. It does not use the test set for tuning.

### Main outputs

- \`artifacts/script_classifier_bert_tiny/model/\` — model and tokenizer for inference.
- \`artifacts/script_classifier_bert_tiny/training_report.json\` — train configuration, tokenizer audit, validation/test metrics, class reports, confusion matrix, mixed-script recall and English false-DEVANAGARI rate.
- \`artifacts/script_classifier_bert_tiny/test_predictions.csv\` — per-example test predictions and probabilities.
- \`artifacts/script_classifier_bert_tiny/tokenizer_audit.json\` — sampled \`[UNK]\` token exposure by script slice.

Inspect \`tokenizer_audit.json\` before trusting results. A high \`[UNK]\` fraction for Devanagari means the base tokenizer represents Hindi poorly; model scores can still be informative for this first experiment, but do not treat them as evidence that lexical content is well represented.

## 3. Benchmark CPU inference

Download/copy the saved model folder to the target Windows machine. From the repository root, install \`torch\`, \`transformers\`, \`pandas\`, and \`numpy\` in that environment, then run:

\`\`\`powershell
python scripts/benchmark_script_classifier_cpu.py --model-dir artifacts/script_classifier_bert_tiny/model --input-csv data/processed/script_classifier/test.csv --samples 1000 --warmup 50 --output artifacts/script_classifier_bert_tiny/cpu_benchmark.json
\`\`\`

For a fair baseline comparison, run the exact same command with \`--model-dir\` pointing to the existing fine-tuned baseline model directory. Keep the same input CSV, sample count, max length, and thread count. The benchmark reports end-to-end per-utterance latency including tokenization, p50/p95/p99, throughput, and model-directory size.

## 4. Interpretation checklist

1. Compare macro-F1, each class's precision/recall/F1, mixed-script recall, and English false-DEVANAGARI rate.
2. Inspect incorrect examples, especially \`Eligibility क्या है?\` and other mixed-script text.
3. Treat this source-derived test score as a first benchmark, not final production accuracy. Evaluate a separate, human-reviewed set of real Sarvam \`codemix\` transcripts. Keep the reserved \`call_data(3).txt\` final evaluation set untouched until model selection is complete.
4. Deploy only if the error profile and target-Windows CPU latency are acceptable.
