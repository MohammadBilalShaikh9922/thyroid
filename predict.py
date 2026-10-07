from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/thyroid_model.pkl")

def predict(patient_data: dict):
    if not MODEL_PATH.exists():
        from train import train
        train()

    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([patient_data])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    classes = model.classes_

    confidence = float(max(probabilities))
    return prediction, confidence, dict(zip(classes, probabilities))
