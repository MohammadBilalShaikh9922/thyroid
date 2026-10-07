import io
from pathlib import Path
import requests
import pandas as pd

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease"
ALLHYPER_URL = f"{BASE}/allhyper.data"
SICK_URL = f"{BASE}/sick-euthyroid.data"

DATA_DIR = Path("data")
OUT = DATA_DIR / "thyroid_multiclass.csv"

FEATURES = [
    "age", "sex",
    "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment",
    "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre",
    "TSH", "T3", "TT4", "T4U", "FTI"
]

def download(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text

def clean_token(x):
    x = str(x).strip().lower()
    return "" if x in {"?", "nan", ""} else x

def parse_allhyper(text):
    rows = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or "|" not in raw:
            continue
        left = raw.split("|", 1)[0]
        p = [clean_token(x) for x in left.split(",")]
        if len(p) < 30:
            continue

        label = p[29].replace(".", "")
        if label == "negative":
            target = "Euthyroid"
        elif any(k in label for k in ["hyperthyroid", "toxic", "goitre"]):
            target = "Hyperthyroid"
        else:
            continue

        # Common clinical/lab attributes shared with the sick-euthyroid file.
        row = [
            p[0], p[1],
            *p[2:14],
            p[17], p[19], p[21], p[23], p[25]
        ]
        if len(row) == len(FEATURES):
            rows.append(row + [target])
    return rows

def parse_sick(text):
    rows = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        p = [clean_token(x) for x in raw.split(",")]
        if len(p) < 27:
            continue

        label = p[0]
        if label == "sick-euthyroid":
            target = "Euthyroid-sick"
        elif label == "negative":
            target = "Euthyroid"
        else:
            continue

        # This file has 12 common binary clinical flags followed by
        # measured-value pairs for TSH, T3, TT4, T4U and FTI.
        row = [
            p[1], p[2],
            *p[3:15],
            p[16], p[18], p[20], p[22], p[24]
        ]
        if len(row) == len(FEATURES):
            rows.append(row + [target])
    return rows

def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading UCI thyroid datasets...")
    hyper_text = download(ALLHYPER_URL)
    sick_text = download(SICK_URL)

    rows = parse_allhyper(hyper_text) + parse_sick(sick_text)
    df = pd.DataFrame(rows, columns=FEATURES + ["target"])

    # Convert numeric columns and binary flags consistently.
    numeric = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in [x for x in FEATURES if x not in numeric]:
        df[c] = df[c].replace({"t": "yes", "f": "no", "y": "yes", "n": "no"})

    # Remove exact duplicate patient records.
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(OUT, index=False)

    print(f"Saved: {OUT}")
    print("\nClass distribution:")
    print(df["target"].value_counts())
    print(f"\nRows: {len(df)}, Features: {len(FEATURES)}")

if __name__ == "__main__":
    main()
