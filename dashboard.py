import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Animal Mind Reader", layout="centered")
st.title("Animal Mind Reader — research prototype")
st.caption("Neural decoding demo. This predicts trained experimental labels; it does not read arbitrary thoughts.")

model_file = st.file_uploader("Upload baseline model (.joblib)", type=["joblib"])
data_file = st.file_uploader("Upload processed trial dataset (.npz)", type=["npz"])

if model_file and data_file:
    bundle = joblib.load(model_file)
    d = np.load(data_file, allow_pickle=True)
    labels = d["labels"]

    trial = st.number_input(
        "Trial index",
        min_value=0,
        max_value=len(d["X"]) - 1,
        value=0,
        step=1,
    )

    x = d["X"][trial].mean(axis=0).reshape(1, -1)
    probs = bundle["model"].predict_proba(x)[0]
    order = np.argsort(probs)[::-1]

    st.subheader("Prediction")
    for i in order:
        st.write(f"**{labels[i]}** — {probs[i] * 100:.1f}%")
