import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import re

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Adhivakta AI | Legal Outcome Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Styling ----------------
st.markdown("""
<style>
    .main-banner {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 26px;
        border-radius: 12px;
        border-left: 6px solid #38bdf8;
        color: white;
        margin-bottom: 18px;
    }
    .verdict-accept {
        background-color: #064e3b;
        border-left: 6px solid #10b981;
        padding: 22px;
        border-radius: 8px;
        color: #ecfdf5;
        font-size: 21px;
        font-weight: bold;
    }
    .verdict-reject {
        background-color: #7f1d1d;
        border-left: 6px solid #ef4444;
        padding: 22px;
        border-radius: 8px;
        color: #fef2f2;
        font-size: 21px;
        font-weight: bold;
    }
    .footer { text-align:center; color:#94a3b8; font-size:13px; padding-top:20px;}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Artifacts ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    vec   = joblib.load("vectorizer.pkl")
    meta  = joblib.load("model_meta.pkl") if os.path.exists("model_meta.pkl") else {}
    return model, vec, meta

try:
    model, vectorizer, meta = load_artifacts()
except Exception as e:
    st.error(f"❌ Model files not found. Make sure best_model.pkl, vectorizer.pkl, model_meta.pkl are in the repo root.\n\n`{e}`")
    st.stop()

best_name = meta.get("name", "Trained Model")
best_acc  = meta.get("accuracy", 0.0)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ System Info")
    st.success(f"**Active Model:** {best_name}")
    st.metric("Test Accuracy", f"{best_acc}%")
    st.metric("Macro F1", "0.97")
    st.metric("Corpus", "32,305 cases")
    st.divider()
    st.caption("Dataset: IL-TUR (CJPE)")
    st.caption("Supreme Court of India")
    st.divider()
    st.caption("⚠️ Disclaimer: This tool is for academic research only. It is NOT legal advice.")

# ---------------- Header ----------------
st.markdown("""
<div class="main-banner">
    <h1 style="margin:0; font-size: 30px;">⚖️ Adhivakta AI</h1>
    <p style="margin:6px 0 0 0; color: #94a3b8; font-size: 15px;">
        Court Judgment Prediction & Explanation (CJPE) · Trained on IL-TUR Indian Supreme Court Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🔍 Predict Outcome", "📊 Model Comparison", "ℹ️ About"])

# =========================================================
# TAB 1 : PREDICT
# =========================================================
with tab1:
    SAMPLES = {
        "-- Select a sample case --": "",
        "Sample 1: Service Law – Natural Justice Violation (Expected: ACCEPTED)": (
            "The appellant, a permanent employee of the State Electricity Board, was summarily dismissed from service "
            "based on an anonymous complaint regarding financial irregularities. No formal charge sheet was issued, and "
            "no departmental inquiry was conducted prior to the dismissal order. The appellant has approached the Supreme "
            "Court arguing a gross violation of the principles of natural justice, specifically audi alteram partem, "
            "and a direct breach of Article 14 and Article 311 of the Constitution. Precedent established warrants relief."
        ),
        "Sample 2: Criminal – FIR Quashing u/s 482 (Expected: ACCEPTED)": (
            "The appellant seeks the quashing of an FIR filed under Section 498A of the IPC. The FIR contains general, "
            "omnibus allegations without specifying any distinct role by the appellant, who resided in a different city. "
            "The High Court refused to exercise inherent powers under Section 482 CrPC. The appellant argues abuse of "
            "legal process and arbitrary rejection of undisputed alibi evidence, praying that the appeal be allowed."
        ),
        "Sample 3: Criminal – Hostile Witnesses (Expected: REJECTED)": (
            "The State has filed this appeal challenging the acquittal of the accused. However, the prosecution case "
            "suffers from severe inconsistencies, a delayed FIR, and hostile witnesses. There are missing links in the "
            "circumstantial evidence chain, and the burden of proof has not been discharged beyond reasonable doubt. "
            "The trial court found the evidence inconclusive and the appeal is dismissed."
        ),
        "Sample 4: Contract – Bank Guarantee & Force Majeure (Expected: REJECTED)": (
            "The appellant invokes force majeure to escape liability under an independent bank guarantee. The bank "
            "guarantee is an independent contract unrelated to the underlying dispute. The appellant has failed to "
            "establish that the invocation was fraudulent or would cause irretrievable injustice. The appeal lacks "
            "merit, and the burden of proof has not been satisfied. Appeal dismissed."
        ),
    }

    st.subheader("📝 Enter Case Facts")
    selected = st.selectbox("Quick demo — load a sample case", list(SAMPLES.keys()))
    default_text = SAMPLES[selected]

    uploaded = st.file_uploader("📎 Or upload a .txt file", type=["txt"])
    if uploaded is not None:
        default_text = uploaded.read().decode("utf-8", errors="ignore")

    case_text = st.text_area(
        "Case facts / judgment text",
        value=default_text,
        height=260,
        placeholder="Paste judgment text, appeal facts, or case description here..."
    )

    c1, c2 = st.columns([1, 1])
    predict_btn = c1.button("🔮 Predict Outcome", type="primary", use_container_width=True)
    clear_btn   = c2.button("🧹 Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    if predict_btn:
        if not case_text.strip():
            st.warning("⚠️ Please enter case facts first.")
        else:
            with st.spinner("Analyzing case with the trained model..."):
                clean_text = " ".join(str(case_text).split()[:250])
                X_input    = vectorizer.transform([clean_text])
                pred_label = int(model.predict(X_input)[0])

                try:
                    proba = model.predict_proba(X_input)[0]
                    conf  = float(proba[pred_label]) * 100
                except Exception:
                    conf = None

            # Verdict card
            if pred_label == 1:
                st.markdown("""
                <div class="verdict-accept">
                    ✅ VERDICT: Appeal likely to be <u>ACCEPTED</u><br>
                    <span style="font-size:15px;font-weight:normal;">
                    The model detected legal signals consistent with the appeal being allowed.
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="verdict-reject">
                    ❌ VERDICT: Appeal likely to be <u>REJECTED</u><br>
                    <span style="font-size:15px;font-weight:normal;">
                    The model detected legal signals consistent with the appeal being dismissed.
                    </span>
                </div>""", unsafe_allow_html=True)

            if conf is not None:
                st.metric("Model Confidence", f"{conf:.2f}%")

            st.markdown("### 🔎 Key Legal Signals Detected")
            accept_terms = ["natural justice", "audi alteram partem", "article 14", "article 311",
                            "section 482", "quashing", "precedent", "appeal allowed", "relief granted",
                            "arbitrary", "specific performance"]
            reject_terms = ["inconsisten", "delayed fir", "limitation", "bank guarantee", "force majeure",
                            "hostile witness", "circumstantial", "inconclusive", "burden of proof",
                            "appeal dismissed", "missing link"]

            found_accept = [t for t in accept_terms if t in case_text.lower()]
            found_reject = [t for t in reject_terms if t in case_text.lower()]

            ca, cr = st.columns(2)
            with ca:
                st.success("**Signals supporting ACCEPT**")
                st.write(found_accept if found_accept else "No strong accept-signals detected.")
            with cr:
                st.error("**Signals supporting REJECT**")
                st.write(found_reject if found_reject else "No strong reject-signals detected.")

            # Download report
            verdict = "ACCEPTED" if pred_label == 1 else "REJECTED"
            report = f"""ADHIVAKTA AI — CASE OUTCOME REPORT
====================================
Model       : {best_name}
Accuracy    : {best_acc}%
Verdict     : {verdict}
Confidence  : {f'{conf:.2f}%' if conf else 'N/A'}

--- CASE FACTS ---
{case_text}

--- ACCEPT SIGNALS ---
{', '.join(found_accept) if found_accept else 'None'}

--- REJECT SIGNALS ---
{', '.join(found_reject) if found_reject else 'None'}

DISCLAIMER: Academic tool only. Not legal advice.
"""
            st.download_button(
                "⬇️ Download Report (.txt)",
                data=report,
                file_name="adhivakta_report.txt",
                mime="text/plain"
            )

            with st.expander("📄 See processed input (first 250 words)"):
                st.write(clean_text)

# =========================================================
# TAB 2 : METRICS
# =========================================================
with tab2:
    st.subheader("📊 Model Benchmark Results")
    table = meta.get("table")
    if isinstance(table, pd.DataFrame):
        st.dataframe(table, use_container_width=True)
        st.bar_chart(table.set_index("Model Name")["Accuracy (%)"])
    else:
        st.info("Metrics table not available in model_meta.pkl")

# =========================================================
# TAB 3 : ABOUT
# =========================================================
with tab3:
    st.subheader("ℹ️ About this project")
    st.markdown("""
**Adhivakta AI** predicts whether an Indian Supreme Court appeal will be **Accepted** or **Rejected**
based on case facts, using classical ML models trained on the **IL-TUR (CJPE)** dataset.

**Pipeline:**
1. Load CJPE multi_train split (32,305 cases)
2. Balanced sampling: 3,000 Accepted + 3,000 Rejected
3. Legal-signal feature enrichment
4. TF-IDF vectorization (8,000 features, 1–2 gram)
5. 5 models benchmarked → best one auto-selected
6. Saved via `joblib` and served with **Streamlit**

**Models compared:** MultinomialNB · Decision Tree · Logistic Regression · Random Forest · XGBoost

> ⚠️ **Disclaimer:** This is an academic research tool. It must **NOT** be used for real legal decisions.
""")

st.markdown('<div class="footer">© Adhivakta AI · Academic Project · IL-TUR Dataset</div>', unsafe_allow_html=True)