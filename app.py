"""Streamlit playground for the fine-tuned BERT-Tiny script classifier."""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(
    page_title="Hinglish Script Classifier",
    page_icon="🔤",
    layout="centered",
)

MODEL_DIR = Path(__file__).parent / "artifacts" / "script_classifier_bert_tiny" / "model"
MAX_LENGTH = 96

st.title("Hinglish Script Classifier")
st.caption("Fine-tuned BERT-Tiny · DEVANAGARI vs LATIN")
st.info(
    "This app reports the dominant writing script, not the spoken language. "
    "For mixed-script text, the final label follows Unicode letter counts. "
    "The BERT prediction and probabilities are shown separately for comparison."
)

def count_script_letters(text: str) -> tuple[int, int]:
    """Count Devanagari and Latin Unicode letters, ignoring punctuation/digits."""
    devanagari = sum(1 for char in text if 0x0900 <= ord(char) <= 0x097F and char.isalpha())
    latin = sum(1 for char in text if (0x0041 <= ord(char) <= 0x005A or 0x0061 <= ord(char) <= 0x007A) and char.isalpha())
    return devanagari, latin


def dominant_script(text: str, model_prediction: str) -> str:
    devanagari, latin = count_script_letters(text)
    # Product rule: any Devanagari in a mixed-script utterance takes priority.
    if devanagari > 0:
        return "DEVANAGARI"
    if latin > 0:
        return "LATIN"
    return "NO_LETTERS"


@st.cache_resource(show_spinner="Loading fine-tuned BERT-Tiny model…")
def load_classifier(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    labels = {
        str(model.config.id2label.get(i, model.config.id2label.get(str(i), ""))).upper()
        for i in range(model.config.num_labels)
    }
    if labels != {"DEVANAGARI", "LATIN"}:
        raise ValueError(f"Unexpected model labels: {labels}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device


@st.cache_resource
def get_model_path():
    return MODEL_DIR


st.markdown("### Try a transcript")
if "utterance_input" not in st.session_state:
    st.session_state["utterance_input"] = ""

examples = st.selectbox(
    "Or choose an example",
    [
        "Choose an example…",
        "Eligibility क्या है?",
        "Mujhe course ki details batao",
        "मुझे course की details बताओ",
        "Please call me tomorrow.",
        "हाँ, ठीक है",
    ],
)
if st.button("Use selected example"):
    if examples != "Choose an example…":
        st.session_state["utterance_input"] = examples

utterance = st.text_area(
    "Transcript text",
    key="utterance_input",
    height=120,
    help="Paste one finalized STT transcript. Mixed-script text is supported.",
)

classify = st.button("Classify script", type="primary", use_container_width=True)

if classify:
    if not utterance.strip():
        st.warning("Enter a transcript first.")
    elif not (MODEL_DIR / "config.json").exists() or not (MODEL_DIR / "model.safetensors").exists():
        st.error(f"Model files not found in: {MODEL_DIR}")
        st.markdown(
            "Make sure the fine-tuned model files are committed under "
            "`artifacts/script_classifier_bert_tiny/model/` on the deployed branch."
        )
    else:
        try:
            with st.spinner("Loading model if needed and classifying…"):
                tokenizer, model, device = load_classifier(str(MODEL_DIR))
                encoded = tokenizer(
                    utterance.strip(),
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.inference_mode():
                    probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latency_ms = (time.perf_counter() - start) * 1000

                values = probabilities.detach().cpu().tolist()
                id2label = model.config.id2label
                scores = {
                    str(id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))).upper(): float(p)
                    for i, p in enumerate(values)
                }
                model_prediction = max(scores, key=scores.get)
                devanagari_count, latin_count = count_script_letters(utterance.strip())
                prediction = dominant_script(utterance.strip(), model_prediction)

            st.subheader("Final prediction")
            c1, c2 = st.columns(2)
            c1.metric("Predicted script", prediction)
            c2.metric("BERT model prediction", model_prediction)
            st.caption(
                f"Unicode letter counts — Devanagari: {devanagari_count}, Latin: {latin_count}. "
                "Any Devanagari letters take priority in mixed-script text; otherwise Latin letters map to LATIN."
            )
            st.metric("Inference latency", f"{latency_ms:.2f} ms")
            st.caption(
                "Latency excludes first-time model loading. Scores are softmax probabilities "
                "and are not guaranteed to be calibrated."
            )
            st.markdown("### Class probabilities")
            st.bar_chart(pd.DataFrame({"Class": list(scores.keys()), "Probability": list(scores.values())}).set_index("Class"))
            st.write(
                "**Interpretation:** "
                + (
                    "Devanagari is the dominant writing script in this text."
                    if prediction == "DEVANAGARI"
                    else "Latin is the dominant writing script in this text."
                    if prediction == "LATIN"
                    else "No Devanagari or Latin letters were found."
                )
            )
            st.code(
                f"Input: {utterance.strip()}\n"
                f"Final prediction: {prediction}\n"
                f"BERT prediction: {model_prediction}\n"
                f"Devanagari letter count: {devanagari_count}\n"
                f"Latin letter count: {latin_count}\n"
                f"Probabilities: {scores}\n"
                f"Inference latency: {latency_ms:.2f} ms",
                language="text",
            )
        except Exception as exc:
            st.error(f"Classification failed: {type(exc).__name__}: {exc}")
            st.exception(exc)

st.divider()
st.markdown("### Model details")
st.write("- Architecture: BERT-Tiny sequence classifier")
st.write("- Labels: DEVANAGARI and LATIN")
st.write("- Maximum token length: 96")
st.write("- Model directory: `artifacts/script_classifier_bert_tiny/model/`")
st.caption(
    "The reported held-out test score was perfect on the current test split, but this does not "
    "guarantee performance on real Sarvam transcripts. Evaluate on consented, anonymized call data "
    "before relying on it for production language routing."
)
