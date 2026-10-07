import json
from pathlib import Path
import pandas as pd
import streamlit as st

from predict import predict
from train import train

st.set_page_config(
    page_title="Thyroid Disorder Classifier",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Thyroid Disorder Multiclass Classification")
st.caption("Machine-learning demonstration using clinical and laboratory test data.")

MODEL = Path("models/thyroid_model.pkl")
METRICS = Path("outputs/metrics.json")
CM = Path("outputs/confusion_matrix.png")

if not MODEL.exists():
    with st.spinner("First-time setup: downloading the UCI dataset and training the model..."):
        train()

st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=40)
sex = st.sidebar.selectbox("Sex", ["F", "M"])

def yn(label):
    return st.sidebar.selectbox(label, ["no", "yes"])

patient = {
    "age": age,
    "sex": sex,
    "on_thyroxine": yn("On thyroxine"),
    "query_on_thyroxine": yn("Query on thyroxine"),
    "on_antithyroid_medication": yn("On antithyroid medication"),
    "sick": yn("Currently sick"),
    "pregnant": yn("Pregnant"),
    "thyroid_surgery": yn("Thyroid surgery"),
    "I131_treatment": yn("I-131 treatment"),
    "query_hypothyroid": yn("Query hypothyroid"),
    "query_hyperthyroid": yn("Query hyperthyroid"),
    "lithium": yn("Lithium"),
    "goitre": yn("Goitre"),
}

st.subheader("Laboratory Results")
c1, c2, c3, c4, c5 = st.columns(5)
patient["TSH"] = c1.number_input("TSH", min_value=0.0, value=1.5, step=0.01)
patient["T3"] = c2.number_input("T3", min_value=0.0, value=1.2, step=0.01)
patient["TT4"] = c3.number_input("TT4", min_value=0.0, value=100.0, step=0.1)
patient["T4U"] = c4.number_input("T4U", min_value=0.0, value=1.0, step=0.01)
patient["FTI"] = c5.number_input("FTI", min_value=0.0, value=100.0, step=0.1)

if st.button("🔍 Predict Thyroid Disorder", type="primary"):
    result, confidence, probabilities = predict(patient)

    st.subheader("Prediction")
    st.success(f"Predicted class: **{result}**")
    st.metric("Model confidence", f"{confidence * 100:.2f}%")

    st.subheader("Class probabilities")
    prob_df = pd.DataFrame({
        "Class": list(probabilities.keys()),
        "Probability": [round(v * 100, 2) for v in probabilities.values()]
    }).sort_values("Probability", ascending=False)
    st.dataframe(prob_df, use_container_width=True, hide_index=True)
    st.bar_chart(prob_df.set_index("Class"))

st.divider()
st.subheader("Model Evaluation")

if METRICS.exists():
    metrics = json.loads(METRICS.read_text())
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    b.metric("Precision", f"{metrics['precision_macro'] * 100:.2f}%")
    c.metric("Recall", f"{metrics['recall_macro'] * 100:.2f}%")
    d.metric("F1-score", f"{metrics['f1_macro'] * 100:.2f}%")

if CM.exists():
    st.image(str(CM), caption="Confusion Matrix", use_container_width=True)

st.info(
    "Educational/research demonstration only. This prediction is not a medical diagnosis "
    "and should not be used to make treatment decisions."
)
