import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import torch
import pickle
import pandas as pd
from src.model import GRURec
from src.sasrec import SASRec
from src.transformer_rec import TransformerRec
from src.predict import predict_next

st.set_page_config(page_title="Next-Action Predictor", page_icon="🛒", layout="wide")

st.title("🛒 Session-Based Next-Action Predictor")
st.markdown("Predict the next item a user will click, using three different deep learning models.")

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "GRU4Rec": {
        "model_path": "src/model.pth",
        "encoder_path": "src/label_encoder.pkl",
        "class": GRURec,
        "hr5": 0.0044, "hr10": 0.0050, "mrr5": 0.0022, "mrr10": 0.0023,
        "desc": "RNN-based model using Gated Recurrent Units. Fast baseline."
    },
    "SASRec": {
        "model_path": "src/sasrec_model.pth",
        "encoder_path": "src/label_encoder.pkl",
        "class": SASRec,
        "hr5": 0.2042, "hr10": 0.2275, "mrr5": 0.1514, "mrr10": 0.1545,
        "desc": "Self-attentive model. Best performer — 20% HR@5 on unseen data."
    },
    "TransformerRec": {
        "model_path": "src/transformer_model.pth",
        "encoder_path": "src/label_encoder.pkl",
        "class": TransformerRec,
        "hr5": 0.1674, "hr10": 0.1930, "mrr5": 0.1302, "mrr10": 0.1334,
        "desc": "Deeper Transformer with 4 heads and 3 layers."
    },
}

@st.cache_resource
def load_model(model_name):
    cfg = MODEL_OPTIONS[model_name]
    with open(cfg["encoder_path"], "rb") as f:
        le = pickle.load(f)
    model = cfg["class"](num_items=len(le.classes_))
    model.load_state_dict(torch.load(cfg["model_path"], map_location="cpu"))
    model.eval()
    return model, le

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox("Choose Model", list(MODEL_OPTIONS.keys()), index=1)
    cfg = MODEL_OPTIONS[selected_model]
    st.info(cfg["desc"])

    st.markdown("---")
    st.header("📊 Model Performance (Test Set)")
    perf_df = pd.DataFrame([
        {"Model": name, "HR@5": v["hr5"], "HR@10": v["hr10"],
         "MRR@5": v["mrr5"], "MRR@10": v["mrr10"]}
        for name, v in MODEL_OPTIONS.items()
    ])
    st.dataframe(perf_df.set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.header("📋 How to use")
    st.markdown("""
    1. Choose a model above
    2. Enter item IDs separated by commas
    3. Adjust K (number of predictions)
    4. Click **Predict**
    """)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    model, le = load_model(selected_model)
    st.success(f"✅ **{selected_model}** loaded — vocab size: {len(le.classes_):,} items")
except Exception as e:
    st.error(f"❌ Could not load {selected_model}: {e}")
    st.stop()

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("📥 Input Session")
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🎲 Load sample session"):
        sample = le.classes_[:5].tolist()
        st.session_state["session_input"] = ", ".join(str(x) for x in sample)

    default_val = st.session_state.get("session_input", "")
    session_input = st.text_input(
        "Item IDs (comma-separated)",
        value=default_val,
        placeholder="e.g. 355908, 248676, 318965"
    )

with col2:
    k = st.slider("Predictions (K)", min_value=1, max_value=10, value=5)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Next Items", type="primary"):
    if not session_input.strip():
        st.warning("Please enter at least one item ID.")
    else:
        try:
            raw_items = [x.strip() for x in session_input.split(",") if x.strip()]
            try:
                items = [int(x) for x in raw_items]
            except ValueError:
                items = raw_items

            known_classes = set(le.classes_.tolist())
            valid_items   = [i for i in items if i in known_classes]
            unknown_items = [i for i in items if i not in known_classes]

            if unknown_items:
                st.warning(f"⚠️ Skipped unknown item IDs: {unknown_items}")

            if len(valid_items) == 0:
                st.error("No valid item IDs found. Try clicking 'Load sample session'.")
            else:
                predictions = predict_next(model, valid_items, le, k=k)

                st.subheader(f"🎯 Top {k} Predictions — {selected_model}")
                st.markdown(f"**Input session:** `{valid_items}`")

                col_left, col_right = st.columns(2)
                with col_left:
                    for rank, item in enumerate(predictions, 1):
                        st.markdown(f"**#{rank}** — Item `{item}`")
                with col_right:
                    result_df = pd.DataFrame({
                        "Rank": list(range(1, len(predictions) + 1)),
                        "Predicted Item ID": predictions
                    })
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)

st.markdown("---")
st.caption("Session-Based Next-Action Predictor | GRU4Rec · SASRec · TransformerRec | RetailRocket Dataset | Trained on 50k sequences · 10 epochs")