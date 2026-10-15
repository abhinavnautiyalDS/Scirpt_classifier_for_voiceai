# Qwen2.5-0.5B Hinglish Language Switch LoRA Adapter

This directory contains the trained LoRA adapter for the four-label language-switch experiment.

## Required inference files

- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja` (when included in the training output)

The base model is `Qwen/Qwen2.5-0.5B-Instruct`. The app loads the base model and attaches this adapter through PEFT.

## Evaluation note

The experiment's provisional test run reported 96.58% accuracy and 0.9427 macro-F1 on 322 automatically generated heuristic labels. These metrics have not been validated on a human-reviewed test set and must not be presented as verified real-world performance.

Do not include `checkpoint-198/`, `checkpoint-297/`, optimizer states, scheduler states, RNG states, or raw call transcripts in this folder. They are not required for inference.

If the adapter files are not present in this directory, the Streamlit app will report that the model is missing.
