"""
Phishing Detection Dashboard — Baseline Models with Explainability
Project 32 — Week 2

Run AFTER baseline_model.py has been run at least once (it needs
best_baseline_model.pkl, tfidf_vectorizer.pkl, and results.csv to exist
in the same folder).

HOW TO RUN:
    pip install streamlit
    streamlit run app.py

This opens a browser tab at http://localhost:8501

------------------------------------------------------------------
HOW THE "REASONS" WORK (read this so it's not a black box to you):

Every model here is trained on TF-IDF features — basically each word in
a message gets a numeric "importance score". When you type a message in,
we look at WHICH words from your message the model actually used, and how
much each one pushed the decision toward "Phishing" or "Legitimate".

- For Logistic Regression / SVM: every word has a learned coefficient
  (a weight). Positive weight = pushes toward phishing, negative = pushes
  toward legitimate. We just sort your message's words by this weight.
- For Naive Bayes: same idea, but using the model's learned log-probabilities
  per class instead of coefficients.
- For Random Forest: it doesn't give per-word weights directly, so we fall
  back to just showing the words in your message with the highest TF-IDF
  score (i.e. the words the model "noticed" most).

This is a simple, honest explainability layer — not the full SHAP-based
explainability you'll build later in the project, but enough to show WHY
a baseline model made its call.
------------------------------------------------------------------
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------
# Page setup + dark theme
# ----------------------------
st.set_page_config(page_title="Phishing Detection Dashboard", layout="centered")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    .reason-box {
        background-color: #1c1f26;
        border-left: 3px solid #444;
        padding: 10px 14px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-family: monospace;
    }
    .verdict-phishing {
        background-color: #3a1414;
        border: 1px solid #8a2e2e;
        padding: 16px;
        border-radius: 6px;
        font-size: 20px;
        font-weight: 600;
        color: #ff6b6b;
    }
    .verdict-legit {
        background-color: #142a1c;
        border: 1px solid #2e8a4f;
        padding: 16px;
        border-radius: 6px;
        font-size: 20px;
        font-weight: 600;
        color: #6bff9a;
    }
</style>
""", unsafe_allow_html=True)

st.title("Phishing Detection — Baseline Model Dashboard")
st.caption("Project 32 | Week 2 | Baseline model testing + explainability")

# ----------------------------
# Load trained artifacts
# ----------------------------
required_files = ["best_baseline_model.pkl", "tfidf_vectorizer.pkl", "results.csv"]
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    st.error(f"Missing files: {missing}. Run 'python baseline_model.py' first in this folder.")
    st.stop()

model = joblib.load("best_baseline_model.pkl")
print("MODEL TYPE:", type(model))
print("HAS COEF:", hasattr(model, "coef_"))

if hasattr(model, "coef_"):
    print("COEF TYPE:", type(model.coef_))
    print("COEF SHAPE:", model.coef_.shape)
vectorizer = joblib.load("tfidf_vectorizer.pkl")
results_df = pd.read_csv("results.csv")
feature_names = np.array(vectorizer.get_feature_names_out())

# ----------------------------
# Model comparison
# ----------------------------
st.subheader("Model Comparison")
st.dataframe(results_df, use_container_width=True)
st.bar_chart(results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]])

best_model_name = results_df.iloc[0]["Model"]
st.write(f"Best performing model so far: **{best_model_name}**")

st.divider()


# ----------------------------
# Explainability logic
# ----------------------------
def get_top_reasons(model, vectorizer, text, top_n=6):

    vec = vectorizer.transform([text])

    nonzero_idx = vec.nonzero()[1]

    if len(nonzero_idx) == 0:
        return []

    feature_names = np.array(vectorizer.get_feature_names_out())

    words = feature_names[nonzero_idx]

    tfidf_scores = np.asarray(
        vec[0, nonzero_idx].todense()
    ).flatten()

    # Logistic Regression explainability
    if hasattr(model, "coef_"):

        coef = np.asarray(model.coef_)

        word_weights = coef[0, nonzero_idx]

        contribution = word_weights * tfidf_scores

    # Naive Bayes explainability
    elif hasattr(model, "feature_log_prob_"):

        log_diff = (
            model.feature_log_prob_[1, nonzero_idx]
            - model.feature_log_prob_[0, nonzero_idx]
        )

        contribution = log_diff * tfidf_scores

    else:
        contribution = tfidf_scores

    ranked = sorted(
        zip(words, contribution),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return ranked[:top_n]

# ----------------------------
# Live test section
# ----------------------------
st.subheader("Test a Message")
user_text = st.text_area(
    "Paste an email / SMS / WhatsApp message:",
    height=120,
    placeholder="Your account has been suspended, click here to verify..."
)

if st.button("Analyze Message", type="primary"):
    if not user_text.strip():
        st.warning("Type something first.")
    else:
        vec = vectorizer.transform([user_text])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None

        if pred == 1:
            confidence = proba[1] if proba is not None else None
            st.markdown(
                f'<div class="verdict-phishing">VERDICT: PHISHING / SPAM'
                + (f' &nbsp;&nbsp;|&nbsp;&nbsp; confidence: {confidence:.1%}' if confidence else '')
                + '</div>', unsafe_allow_html=True
            )
        else:
            confidence = proba[0] if proba is not None else None
            st.markdown(
                f'<div class="verdict-legit">VERDICT: LEGITIMATE'
                + (f' &nbsp;&nbsp;|&nbsp;&nbsp; confidence: {confidence:.1%}' if confidence else '')
                + '</div>', unsafe_allow_html=True
            )

        st.write("")
        st.markdown("**Why this verdict — top contributing words:**")

        reasons = get_top_reasons(model, vectorizer, user_text)
        if not reasons:
            st.write("No recognizable words from training vocabulary found in this message.")
        else:
            for word, score in reasons:
                direction = "→ phishing signal" if score > 0 else "→ legitimate signal"
                if not hasattr(model, "coef_") and not hasattr(model, "feature_log_prob_"):
                    direction = "→ high-weight term (Random Forest has no per-word direction)"
                st.markdown(
                    f'<div class="reason-box"><b>{word}</b> &nbsp; (score: {score:.3f}) &nbsp; {direction}</div>',
                    unsafe_allow_html=True
                )

st.divider()
st.caption(
    "Results above reflect whatever dataset baseline_model.py was last trained on "
    "(dummy data today, real dataset once it's plugged in tomorrow)."
)