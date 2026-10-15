# Data and annotation guide

Do not put raw call recordings, raw transcripts, phone numbers, names, or other personally identifying information in this public repository. Keep private source data in controlled storage. Only use data you have permission to process.

## CSV schema

Required columns: text,label. Recommended columns: call_id,reviewed,annotator,notes.

Labels are case-sensitive:

- HINGLISH: Hindi in Devanagari or Roman/Latin script, including meaningful Hindi-English code-mixing. Examples: Mujhe details WhatsApp kar dena; मुझे details भेज दीजिए.
- ENGLISH: meaningful English. Example: Please send me the details tomorrow.
- NEUTRAL: language-agnostic short acknowledgements/greetings/fillers or utterances too short to provide evidence. Examples: hello, okay, yes, bye.
- UNCERTAIN: unsupported language, unintelligible/ambiguous transcription, or insufficient evidence to choose a supported language.

## Annotation rules

1. Label caller turns only; never label the voice agent's own responses as caller language.
2. Preserve what the caller said; do not translate Romanized Hindi into English.
3. Distinguish English from Hindi written in Latin script by language, not alphabet alone.
4. For code-mixing, use HINGLISH when Hindi grammar/lexical content is meaningfully present alongside English.
5. A lone English loanword inside an otherwise Hindi utterance does not make it English.
6. Prefer NEUTRAL for generic short phrases that should not trigger language switching. Use UNCERTAIN for unclear or unsupported content.
7. Keep turns from the same call/speaker group in a single split.
8. Mark reviewed=true only after a human has checked the text and label. Auto labels are provisional.

Include diverse callers, accents, spelling variations, Devanagari and Latin-script Hindi, natural code-switching, English, neutral phrases, unsupported languages, noisy ASR, and ambiguous examples. Avoid one caller or campaign dominating the dataset. Keep the human-reviewed test set locked before model/prompt iteration.
