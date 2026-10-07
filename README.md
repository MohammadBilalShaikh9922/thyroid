# Thyroid Disorder Multiclass Classification

A machine-learning project that predicts three thyroid-related classes:

1. **Hyperthyroid**
2. **Euthyroid-sick**
3. **Euthyroid**

The project uses clinical and laboratory features and a Random Forest classifier.

## Dataset

The project uses the publicly available **UCI Thyroid Disease** data repository.

It automatically downloads:

- `allhyper.data`
- `sick-euthyroid.data`

from the UCI repository when training is started.

The project constructs a common feature set from age, sex, clinical indicators, TSH, T3, TT4, T4U and FTI.

UCI source:
https://archive.ics.uci.edu/dataset/102/thyroid+disease

## Machine Learning Workflow

1. Download public data
2. Clean missing values
3. Convert clinical indicators
4. Combine the relevant classes
5. Stratified 80/20 train-test split
6. Impute missing numerical/categorical values
7. One-hot encode categorical features
8. Train Random Forest
9. Evaluate with:
   - Accuracy
   - Macro Precision
   - Macro Recall
   - Macro F1-score
   - Confusion Matrix
10. Save the trained model with Joblib

## Run Locally

### Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all project files.
3. Go to https://share.streamlit.io/
4. Sign in with GitHub.
5. Select your repository.
6. Select `app.py`.
7. Deploy.

The app downloads the dataset and trains the model automatically on first startup.

## Alternative Deployment

You can also deploy the Streamlit app on Render using a Python web service.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

## Important

This is an academic machine-learning project. It is not a medical diagnostic system.
