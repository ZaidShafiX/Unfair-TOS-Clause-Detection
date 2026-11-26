# Streamlit web app: Unfair Terms of Service (ToS) Clause Detection
#
# Run locally:   streamlit run app.py
# Deploy:        push app.py, requirements.txt, tfidf_vectorizer.pkl and best_model.pkl to a GitHub repo
#                and deploy it on Streamlit Community Cloud (or a Hugging Face Space with the Streamlit SDK).
#
# The preprocessing below is an exact copy of the notebook (Step 3), because the saved TF-IDF vectorizer
# and model were trained on text cleaned this way.

import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")

# ----------------------------------------------------------------------------- preprocessing (same as notebook)
LEGAL_NEGATIONS = {"no", "not", "nor", "never", "without", "cannot", "neither", "none"}
LEGAL_MODALS = {
    "may", "must", "can", "should", "will", "any", "all", "only", "except",
    "upon", "under", "at", "whether", "every", "before", "after", "within",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - (LEGAL_NEGATIONS | LEGAL_MODALS)


def normalize_contractions(text):
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\bcan't\b", "cannot", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)
    text = re.sub(r"n't\b", " not", text)
    return text


def remove_symbols_and_numbers(text):
    return re.sub(r"[^a-zA-Z\s]", " ", str(text))


def remove_stopwords(text):
    return " ".join(word for word in text.split() if word not in STOP_WORDS)


def clean_text(text):
    text = str(text).lower()
    text = normalize_contractions(text)
    text = remove_symbols_and_numbers(text)
    return remove_stopwords(text)


# ----------------------------------------------------------------------------- prediction (same as notebook)
def unfair_probability(model, X):
    unfair_idx = list(model.classes_).index("Unfair")
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, unfair_idx]
    margin = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-margin))


def predict_clause(clause_text, vectorizer, model):
    # Returns (label, confidence in that label)
    vec = vectorizer.transform([clean_text(clause_text)])
    label = model.predict(vec)[0]
    p_unfair = float(unfair_probability(model, vec)[0])
    return label, (p_unfair if label == "Unfair" else 1.0 - p_unfair)


def split_sentences(document_text):
    parts = re.split(r"(?<=[.!?])\s+", document_text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def scan_document(document_text, vectorizer, model):
    rows = []
    for sentence in split_sentences(document_text):
        label, confidence = predict_clause(sentence, vectorizer, model)
        rows.append({"Clause": sentence, "Prediction": label, "Confidence": round(confidence, 3)})
    return pd.DataFrame(rows, columns=["Clause", "Prediction", "Confidence"])


# ----------------------------------------------------------------------------- web app
@st.cache_resource
def load_artifacts():
    return joblib.load(VECTORIZER_PATH), joblib.load(MODEL_PATH)


EXAMPLE_TOS = (
    "Welcome to QuickApp. You must be at least 16 years old to create an account. "
    "We reserve the right to terminate your account at any time without notice or reason. "
    "You can update your profile picture and bio from the settings page. "
    "Under no circumstances shall QuickApp be liable for any loss of data or indirect damages. "
    "All legal disputes shall be settled exclusively by binding arbitration in Delaware."
)


def main():
    st.set_page_config(page_title="Unfair ToS Clause Detector", page_icon="⚖️", layout="centered")
    st.title("⚖️ Unfair Terms of Service Clause Detector")
    st.write(
        "Paste a clause (or a whole Terms of Service section) and the model flags clauses that look "
        "**unfair to the consumer** (e.g. unilateral termination, forced arbitration, liability waivers)."
    )

    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH)):
        st.error("Model files not found. Put 'tfidf_vectorizer.pkl' and 'best_model.pkl' next to app.py.")
        return
    vectorizer, model = load_artifacts()

    tab_single, tab_scan = st.tabs(["Single clause", "Full ToS scanner"])

    with tab_single:
        clause = st.text_area("ToS clause", height=120, key="single",
                              placeholder="We may terminate your account at any time without notice.")
        if st.button("Check clause", key="btn_single"):
            if not clause.strip():
                st.warning("Please enter a clause first.")
            else:
                label, confidence = predict_clause(clause, vectorizer, model)
                if label == "Unfair":
                    st.error(f"UNFAIR  (confidence {confidence:.1%})")
                else:
                    st.success(f"FAIR  (confidence {confidence:.1%})")

    with tab_scan:
        document = st.text_area("Terms of Service text", height=220, value=EXAMPLE_TOS, key="scan")
        if st.button("Scan document", key="btn_scan"):
            result = scan_document(document, vectorizer, model)
            if result.empty:
                st.warning("No sentences found to scan.")
            else:
                n_unfair = int((result["Prediction"] == "Unfair").sum())
                st.metric("Flagged as unfair", f"{n_unfair} of {len(result)} clauses")
                shown = result.copy()
                shown.insert(0, "Flag", shown["Prediction"].map({"Unfair": "🚩", "Fair": "✅"}))
                st.dataframe(shown, hide_index=True)

    st.caption("Model: TF-IDF (1-2 grams) + classical ML classifier trained on the LexGLUE UNFAIR-ToS dataset. "
               "This tool is a screening aid, not legal advice.")


if __name__ == "__main__":
    main()
