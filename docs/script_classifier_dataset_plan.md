# Script Classification Dataset Plan

## Objective

Build a two-label script classifier for the Sarvam voice pipeline. The model predicts only:
- `DEVANAGARI`
- `LATIN`

This is script classification, not response-language classification or language switching. The model's input is `text`; target is `script`.

## Selected sources and extraction policy

### Trilingual Hinglish Corpus
Source: https://huggingface.co/datasets/debajyotimaz/trilingual-hinglish-corpus
- Use `hindi` as a DEVANAGARI baseline candidate, after checking actual Unicode script.
- Use `english` as a LATIN baseline candidate, after checking actual Unicode script.
- Exclude the Romanized `hinglish` column.
- License reported as CC BY-NC-SA 4.0. Do not assume it is suitable for commercial production.

### HinGE
Source: https://huggingface.co/datasets/LingoIITGN/HinGE
- Use `Hindi` as a DEVANAGARI baseline candidate, after checking actual Unicode script.
- Use `English` as a LATIN baseline candidate, after checking actual Unicode script.
- Exclude Human-generated Hinglish, WAC, PAC and other Romanized/generated Hinglish columns from this plan.
- Dataset card reports CC BY 4.0; verify the exact distributed version and attribution requirements.

### COMI-LINGUA
Source: https://github.com/lingo-iitgn/COMI-LINGUA
- Primary source for genuine mixed-script Hindi-English sentences.
- Inspect the exact task schema before extraction.
- Reconstruct sentences only from tokens belonging to the same original sentence and in verified original order.
- Remove annotation tags only after correct reconstruction.
- Keep candidates containing both Devanagari and Latin, then validate language tags/context to confirm genuine Hindi-English code-mixing.
- Label verified mixed-script Hinglish as DEVANAGARI.
- Exclude Roman-only Hinglish and English-only sentences from this subset.
- Resolve the license discrepancy between the repository LICENSE file and dataset documentation before commercial use.

## Target data format

Final training CSV has exactly two columns:
- `text`
- `script`

During extraction and auditing, preserve source dataset, source row ID, original sentence/group ID, source type and annotation metadata in intermediate files/reports. Do not invent original group IDs when unavailable.

## Cleaning and split requirements

1. Inspect actual schemas and representative rows; do not infer scripts from column names.
2. Normalize Unicode to NFC and whitespace conservatively.
3. Do not label using character counts alone.
4. Remove exact duplicates while retaining provenance in intermediate data.
5. Identify aligned variants and near-duplicates; group them before splitting.
6. Keep train/validation/test leakage-resistant. Balance or oversample training only, if needed.
7. Do not train until data labels, source counts, class counts and split logic have been reviewed.
8. Keep real call transcripts and sensitive data out of this public repository.
9. Evaluate separately on pure Devanagari Hindi, pure Latin English, mixed-script Hinglish and real Sarvam codemix transcripts.

## Status

Dataset schema inspection and extraction pipeline are in progress. Actual post-filter counts are unknown until the source files are downloaded and audited. No model training is authorized by this plan yet.
