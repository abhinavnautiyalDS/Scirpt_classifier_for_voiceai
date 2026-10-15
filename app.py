"""Streamlit playground for manually testing a fine-tuned language-switch model."""
import os
import streamlit as st

st.set_page_config(page_title="Hinglish Language Switch LLM", page_icon="🗣️")
st.title("Hinglish Language Switch LLM")
st.caption("Manual test UI. Predictions are candidates, not guaranteed ground truth.")

model_path = os.environ.get("HINGLISH_MODEL_PATH", "artifacts/qwen-hinglish-lora")
base_model = os.environ.get("HINGLISH_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
text = st.text_area("Caller utterance", placeholder="Mujhe details WhatsApp kar dena", height=120)
st.selectbox("Current agent language (context only)", ["Hinglish", "English"])
run = st.button("Classify", type="primary")

if run:
    if not text.strip():
        st.warning("Enter an utterance first.")
    elif not os.path.isdir(model_path):
        st.error(f"Model adapter not found at {model_path}. Train a model or set HINGLISH_MODEL_PATH.")
    else:
        from src.predict import LanguageSwitchClassifier

        @st.cache_resource
        def load_classifier(path, base):
            return LanguageSwitchClassifier(path, base)

        with st.spinner("Classifying…"):
            result = load_classifier(model_path, base_model).classify(text)
        st.subheader("Result")
        st.metric("Predicted label", result.label)
        st.write("Switch candidate:", "Yes — still requires controller checks" if result.switch_allowed else "No")
        st.caption(result.reason)

st.divider()
st.info("Use a human-reviewed, held-out test set and benchmark on the target CPU. No accuracy claim is made by this playground.")
