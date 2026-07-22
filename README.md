# AI-Powered Disease Prediction & Clinical Decision Support System

A prototype web application that combines machine learning risk models with real-time physiological vitals to provide early-warning disease risk estimates and basic clinical decision support. Built with **XGBoost** for prediction and **Gradio** for the interactive interface.

> ⚠️ **Disclaimer:** This is a research/educational prototype, not a certified medical device. Risk estimates are approximations based on publicly available datasets and should never be used as a substitute for professional medical diagnosis. See [Known Limitations](#known-limitations) below.

---

## Features

- Predicts risk for **four conditions**: Diabetes, Hypertension, Stroke, and Heart Failure
- Trains a separate XGBoost classifier per condition from its own dataset
- Accepts patient history (BMI, cholesterol, smoking status, activity level, etc.) through an interactive form
- Accepts real-time vitals (heart rate, SpO2) and flags abnormal readings
- Generates a formatted clinical report with risk percentages, vital alerts, and rule-based recommendations
- Gracefully degrades if a dataset is missing (reports "N/A" instead of crashing)

---

## How It Works

### 1. Data & Model Training
On startup, the script looks for three CSV files in the working directory and trains a model for each available one:

| Dataset File | Condition(s) Predicted | Target Column |
|---|---|---|
| `binary_health_indicators_BRFSS2015.csv` | Diabetes, Hypertension | `Diabetes_binary`, `HighBP` |
| `healthcare-dataset-stroke-data.csv` | Stroke | `stroke` |
| `heart.csv` | Heart Failure | `HeartDisease` / `target` (auto-detected) |

Each model is an `XGBClassifier` (100 estimators, max depth 4, learning rate 0.05), and each dataset's features are standardized with a dedicated `StandardScaler`.

### 2. Prediction Engine
The `predict_clinical_risks()` function:
1. Builds the correct feature vector for each trained model
2. Scales inputs using the matching scaler
3. Outputs a probability (%) for each disease
4. Checks real-time heart rate and SpO2 against normal thresholds
5. Applies simple rule-based logic to suggest follow-up actions (e.g., HbA1c test, cardiology referral)
6. Returns everything as a formatted Markdown report

### 3. User Interface
A two-column Gradio layout:
- **Left column** — Patient medical history (age group, blood pressure, cholesterol, BMI, smoking, activity, general/mental/physical health, education, income)
- **Right column** — Real-time vitals (heart rate, SpO2) and a "Run Analysis" button that displays the generated report

---

## Requirements

```
pandas
numpy
xgboost
scikit-learn
gradio
```

Install with:
```bash
pip install pandas numpy xgboost scikit-learn gradio
```

## Setup & Usage

1. Place the following dataset files in the same directory as the script (any subset can be present — missing files simply disable that condition's prediction):
   - `binary_health_indicators_BRFSS2015.csv`
   - `healthcare-dataset-stroke-data.csv`
   - `heart.csv`
2. Run the script:
   ```bash
   python app.py
   ```
3. Gradio will launch a local interface (and a public shareable link, since `share=True`) where you can enter patient data and click **Run Analysis** to view the report.

---

## Known Limitations

These are important to understand before relying on any output from this app:

- **Heart Failure model input is incomplete.** The prediction only sets the age feature; all other model features are left at zero, meaning predictions are effectively age-only despite the model being trained on many features. Risk scores for this condition should be treated as unreliable.
- **Stroke model uses a hardcoded glucose value** (100.0) instead of an actual patient reading, limiting prediction accuracy.
- **Age is approximated** from a UI dropdown age *group* (`age * 5 + 15`) rather than collected as an exact value, which introduces error into both the Stroke and Heart Failure predictions.
- **Feature-order assumption**: the Heart Failure feature vector assumes the model's first scaled feature is age, which depends on column ordering in `heart.csv` and isn't guaranteed.
- No cross-validation, calibration, or external validation of model outputs is performed.
- Recommendations are simple threshold-based rules, not clinically validated decision logic.

## Suggested Improvements

- Collect real age and glucose level directly in the UI instead of approximating them
- Fully populate the Heart Failure feature vector with actual patient inputs, mapped to the correct columns in `heart.csv`
- Add model evaluation metrics (AUC, calibration curves) to the training output
- Validate against a held-out clinical dataset before any real-world use
