from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from prepare_data import main as prepare_data

DATA_PATH = Path("data/thyroid_multiclass.csv")
MODEL_PATH = Path("models/thyroid_model.pkl")
METRICS_PATH = Path("outputs/metrics.json")
CM_PATH = Path("outputs/confusion_matrix.png")

FEATURES = [
    "age", "sex",
    "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment",
    "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre",
    "TSH", "T3", "TT4", "T4U", "FTI"
]

NUMERIC = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]

def train():
    if not DATA_PATH.exists():
        prepare_data()

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = df["target"].copy()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL)
    ])

    model = RandomForestClassifier(
        n_estimators=350,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_test, pred, output_dict=True, zero_division=0
        )
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    labels = ["Euthyroid", "Euthyroid-sick", "Hyperthyroid"]
    cm = confusion_matrix(y_test, pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title("Thyroid Disorder - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=160)
    plt.close(fig)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall   : {metrics['recall_macro']:.4f}")
    print(f"F1-score : {metrics['f1_macro']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred, zero_division=0))
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
