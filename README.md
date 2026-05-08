# Personality Type Classifier (Introvert vs. Extrovert)

Technical assignment for Decryptogen AI/ML Intern selection — Round 02.
A Stacking Ensemble model trained on behavioral data, served via a FastAPI endpoint deployed on Hugging Face Spaces.

**Live API:** `https://slmaari-personality-classifier.hf.space/predict`  
**Interactive Docs:** `https://slmaari-personality-classifier.hf.space/docs`

---

## API Usage

### Endpoint

```
POST https://slmaari-personality-classifier.hf.space/predict
```

### Request Format

```json
{
  "Time_spent_Alone": 5,
  "Stage_fear": "No",
  "Social_event_attendance": 3,
  "Going_outside": 4,
  "Drained_after_socializing": "No",
  "Friends_circle_size": 6,
  "Post_frequency": 4
}
```

### Response Format

```json
{
  "prediction": "Extrovert",
  "confidence": 0.9981,
  "input_received": { ... }
}
```

### Sample cURL Request

```bash
curl -X POST "https://slmaari-personality-classifier.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time_spent_Alone": 5,
    "Stage_fear": "No",
    "Social_event_attendance": 3,
    "Going_outside": 4,
    "Drained_after_socializing": "No",
    "Friends_circle_size": 6,
    "Post_frequency": 4
  }'
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `Time_spent_Alone` | float | Hours spent alone per day (0-11) |
| `Stage_fear` | string | "Yes" or "No" |
| `Social_event_attendance` | float | Social events per month (0-10) |
| `Going_outside` | float | Times going outside per week (0-7) |
| `Drained_after_socializing` | string | "Yes" or "No" |
| `Friends_circle_size` | float | Number of close friends (0-15) |
| `Post_frequency` | float | Social media posts per week (0-10) |

---

## Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | 91.72% |
| ROC-AUC | 0.9493 |
| Model Type | Stacking Ensemble |
| Base Learners | LightGBM + XGBoost + Random Forest + SVM |
| Meta-Learner | Logistic Regression |
| Dataset Size | 2,900 samples, 7 features |

Benchmarked against 6 models (Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM, Logistic Regression). The Stacking Ensemble was selected for best generalization.

---

## Engineering Assumptions

**Missing Data (~3-4% of rows):** Used KNN Imputation (k=5) instead of mean imputation. Since features are correlated — a person with high `Social_event_attendance` likely also has a high `Friends_circle_size` — neighbors carry more signal than a flat average.

**Categorical Encoding:** `Stage_fear` and `Drained_after_socializing` are binary Yes/No fields. Encoded with a `LabelEncoder` per column, saved as a pickle dict (`label_encoders.pkl`) to guarantee identical mapping between training and the live API.

**Feature Engineering:** Five composite features were created: `social_score`, `introvert_score`, `social_vs_alone_ratio`, `drained_x_alone`, and `friends_x_events`. These ranked in the top predictors by feature importance and improved cross-validation accuracy by approximately 2%.

**Model Choice:** A single LightGBM model achieved 89.31% after hyperparameter tuning. A Stacking Ensemble combining four diverse base learners under a Logistic Regression meta-learner improved this to 91.72%. A neural network was not used — the dataset size (~2,900 samples) is too small and would likely overfit.

---

## Repo Structure

```
personality-classifier/
├── app/
│   ├── main.py              # FastAPI entry point, input/output schemas
│   ├── model_utils.py       # Full inference pipeline (load artifacts, predict)
│   ├── model.pkl            # Trained Stacking Ensemble
│   ├── label_encoders.pkl   # Per-column LabelEncoders for Yes/No fields
│   ├── knn_imputer.pkl      # KNN imputer fitted on training data
│   ├── feature_cols.pkl     # Ordered feature list (prevents column mismatch)
│   └── target_encoder.pkl   # Maps 0/1 back to Introvert/Extrovert
├── notebooks/
│   └── training.ipynb       # Data cleaning, EDA, feature engineering, training
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/SLMaari/personality-classifier.git
cd personality-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place .pkl artifact files in app/

# 4. Run locally
uvicorn app.main:app --reload

# 5. Open interactive docs at http://localhost:8000/docs
```

---

## Deployment

Deployed on **Hugging Face Spaces** using a FastAPI Docker environment.

Start command: `uvicorn app.main:app --host 0.0.0.0 --port 7860`