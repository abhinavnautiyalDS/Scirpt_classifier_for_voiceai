"""Streamlit playground for the fine-tuned Hinglish language-switch adapter."""
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Hinglish Language Switch LLM", page_icon="🗣️", layout="centered")

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "qwen-hinglish-lora"
model_path = Path(os.environ.get("HINGLISH_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
base_model = os.environ.get("HINGLISH_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

st.title("Hinglish Language Switch LLM")
st.caption("Experimental four-class classifier for voice-agent transcript snippets.")

st.info(
    "Labels: HINGLISH, ENGLISH, NEUTRAL, UNCERTAIN. "
    "The classifier proposes a label; a separate policy decides whether to switch languages."
)

@st.cache_resource(show_spinner="Loading Qwen base model and LoRA adapter…")
def load_classifier(path: str, base: str):
    from src.predict import LanguageSwitchClassifier
    return LanguageSwitchClassifier(path, base)

utterance = st.text_area(
    "Caller utterance",
    placeholder="Mujhe details WhatsApp kar dena",
    height=120,
)
current_language = st.selectbox("Current agent language", ["Hinglish", "English"])
run = st.button("Classify utterance", type="primary", use_container_width=True)

if run:
    if not utterance.strip():
        st.warning("Enter a caller utterance first.")
    elif not (model_path / "adapter_config.json").exists():
        st.error(f"LoRA adapter files were not found at: {model_path}")
        st.markdown(
            "Add the extracted adapter files to models/qwen-hinglish-lora/ in the repository "
            "or set HINGLISH_MODEL_PATH to the adapter directory. Required files include "
            "adapter_config.json and adapter_model.safetensors."
        )
    else:
        try:
            with st.spinner("Classifying…"):
                result = load_classifier(str(model_path), base_model).classify(
                    utterance, current_language=current_language
                )
            st.subheader("Prediction")
            st.metric("Predicted label", result["label"])
            st.write(
                "**Switch candidate:**",
                "Yes — apply your switching-controller checks"
                if result["switch_candidate"] else "No",
            )
            st.caption(f"Policy reason: {result['reason']}")
        except Exception as exc:
            st.error(f"Classification failed: {exc}")
            st.exception(exc)

st.divider()
st.caption(
    "Experimental result only. Earlier evaluation reached 96.58% accuracy on 322 provisional "
    "heuristic-labeled samples; this is not human-reviewed real-world accuracy. "
    "Latency on Streamlit Cloud will differ from the Colab T4 benchmark."
)
