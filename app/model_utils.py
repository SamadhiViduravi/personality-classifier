

import os
import joblib
import pandas as pd
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))

def _load(filename):
    path = os.path.join(_HERE, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            "Run the Colab notebook first and place all .pkl files in app/"
        )
    return joblib.load(path)



print("Loading model artifacts...")
_model          = _load("model.pkl")
_label_encoders = _load("label_encoders.pkl")   # dict: {col: LabelEncoder}
_knn_imputer    = _load("knn_imputer.pkl")
_feature_cols   = _load("feature_cols.pkl")      # ordered list of feature names
_target_encoder = _load("target_encoder.pkl")
print(f"Model loaded: {type(_model).__name__}")

# Columns that go through the KNN imputer
_BASE_COLS = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]

# Valid values for binary columns
_VALID_BINARY = {"yes", "no"}


def _validate_binary(value: str, col_name: str) -> str:
    """Normalise and validate a Yes/No field."""
    v = str(value).strip().capitalize()
    if v not in ("Yes", "No"):
        raise ValueError(
            f"Invalid value '{value}' for '{col_name}'. Expected 'Yes' or 'No'."
        )
    return v


def predict(raw_input: dict) -> dict:

    #Step 1: Validate binary fields
    binary_cols = list(_label_encoders.keys())
    for col in binary_cols:
        raw_input[col] = _validate_binary(raw_input[col], col)

    #Step 2: Build single-row DataFrame
    row = pd.DataFrame([raw_input])

    #Step 3: Label-encode Yes/No columns
    for col, le in _label_encoders.items():
        row[col] = le.transform(row[col].astype(str))

    row = row.astype(float)

    #Step 4: KNN Impute (handles any NaN from optional fields)
    row[_BASE_COLS] = _knn_imputer.transform(row[_BASE_COLS])

    #Step 5: Feature engineering (must be IDENTICAL to training)
    row["social_score"] = (
        row["Social_event_attendance"]
        + row["Going_outside"]
        + row["Friends_circle_size"]
        + row["Post_frequency"]
    )
    row["introvert_score"] = (
        row["Time_spent_Alone"]
        + row["Drained_after_socializing"]
        + row["Stage_fear"]
    )
    row["social_vs_alone_ratio"] = (
        row["Social_event_attendance"] / (row["Time_spent_Alone"] + 1)
    )
    row["drained_x_alone"] = (
        row["Drained_after_socializing"] * row["Time_spent_Alone"]
    )
    row["friends_x_events"] = (
        row["Friends_circle_size"] * row["Social_event_attendance"]
    )

    #Step 6: Align column order to what the model was trained on ────────────
    X = row[_feature_cols].values  # shape (1, n_features)

    #Step 7: Predict ────────────────────────────────────────────────────────
    pred_idx   = _model.predict(X)[0]
    pred_proba = _model.predict_proba(X)[0]
    label      = _target_encoder.inverse_transform([pred_idx])[0]
    confidence = float(pred_proba.max())

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
    }
