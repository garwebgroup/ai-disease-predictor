import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import gradio as gr
import warnings
import textwrap

warnings.filterwarnings('ignore')

print("Loading and preparing datasets...")

# 1. DATASET FILE DEFINITIONS
files = [
    "binary_health_indicators_BRFSS2015.csv",
    "healthcare-dataset-stroke-data.csv",
    "heart.csv"
]

# 2. INDIVIDUAL DATASET PROCESSING & MODEL TRAINING
models = {}
scalers = {}

# --- A. DIABETES & HYPERTENSION (from BRFSS2015 dataset) ---
if os.path.exists(files[0]):
    df_brfss = pd.read_csv(files[0])
    
    brfss_features = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                      'PhysActivity', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 
                      'Sex', 'Age', 'Education', 'Income']
    
    # 1. Diabetes Model
    scaler_dia = StandardScaler()
    X_dia = scaler_dia.fit_transform(df_brfss[brfss_features])
    y_dia = df_brfss['Diabetes_binary']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_dia, y_dia, test_size=0.2, random_state=42)
    model_dia = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model_dia.fit(X_tr, y_tr)
    
    models["Diabetes"] = model_dia
    scalers["Diabetes"] = scaler_dia
    
    # 2. Hypertension Model
    scaler_hyp = StandardScaler()
    X_hyp = scaler_hyp.fit_transform(df_brfss[brfss_features])
    y_hyp = df_brfss['HighBP']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_hyp, y_hyp, test_size=0.2, random_state=42)
    model_hyp = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model_hyp.fit(X_tr, y_tr)
    
    models["Hypertension"] = model_hyp
    scalers["Hypertension"] = scaler_hyp

# --- B. STROKE MODEL (from healthcare-dataset-stroke-data.csv) ---
if os.path.exists(files[1]):
    df_stroke = pd.read_csv(files[1])
    
    df_stroke['bmi'].fillna(df_stroke['bmi'].median(), inplace=True)
    df_stroke['gender'] = df_stroke['gender'].map({'Female': 0, 'Male': 1, 'Other': 0}).fillna(0)
    
    stroke_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender']
    
    scaler_stroke = StandardScaler()
    X_str = scaler_stroke.fit_transform(df_stroke[stroke_features])
    y_str = df_stroke['stroke']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_str, y_str, test_size=0.2, random_state=42)
    model_stroke = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model_stroke.fit(X_tr, y_tr)
    
    models["Stroke"] = model_stroke
    scalers["Stroke"] = scaler_stroke

# --- C. HEART FAILURE MODEL (from heart.csv) ---
if os.path.exists(files[2]):
    df_heart = pd.read_csv(files[2])
    
    target_col = 'HeartDisease' if 'HeartDisease' in df_heart.columns else ('target' if 'target' in df_heart.columns else df_heart.columns[-1])
    
    heart_features = [c for c in df_heart.columns if c != target_col and pd.api.types.is_numeric_dtype(df_heart[c])]
    
    scaler_heart = StandardScaler()
    X_hrt = scaler_heart.fit_transform(df_heart[heart_features])
    y_hrt = df_heart[target_col]
    
    # 80/20 split to prevent overfitting
    # If you train your models on 100% of the dataset, the algorithm might simply "memorize" the specific patients in your file rather than learning the underlying medical patterns.
    X_tr, X_te, y_tr, y_te = train_test_split(X_hrt, y_hrt, test_size=0.2, random_state=42)
    model_heart = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model_heart.fit(X_tr, y_tr)
    
    models["Heart Failure"] = model_heart
    scalers["Heart Failure"] = scaler_heart

print(f"Models successfully trained for: {list(models.keys())}")

# 3. CLINICAL DECISION SUPPORT & PREDICTION ENGINE
def predict_clinical_risks(high_bp, high_chol, bmi, smoker, stroke_hist, heart_hist, 
                           phys_activity, gen_hlth, ment_hlth, phys_hlth, diff_walk, 
                           sex, age, education, income, real_time_hr, real_time_spo2, name):
    if not name or name.strip() == "":
        return "⚠️ **Error**: Please enter the patient's name before running the analysis."
    
    results = {}
    approx_age = age * 5 + 17  # Approximate baseline age in years from BRFSS age scale
    edu_scaled = education + 1   # Map Gradio 0-3 dropdown scale to 1-4 baseline scale

    # Diabetes Prediction
    if "Diabetes" in models:
        inp_dia = np.array([[high_bp, high_chol, bmi, smoker, stroke_hist, heart_hist, 
                             phys_activity, gen_hlth, ment_hlth, phys_hlth, diff_walk, 
                             sex, age, edu_scaled, income]])
        scaled_dia = scalers["Diabetes"].transform(inp_dia)
        results["Diabetes"] = round(models["Diabetes"].predict_proba(scaled_dia)[0][1] * 100, 2)
    else:
        results["Diabetes"] = "Dataset Not Available"

    # Hypertension Prediction
    if "Hypertension" in models:
        inp_hyp = np.array([[high_bp, high_chol, bmi, smoker, stroke_hist, heart_hist, 
                             phys_activity, gen_hlth, ment_hlth, phys_hlth, diff_walk, 
                             sex, age, edu_scaled, income]])
        scaled_hyp = scalers["Hypertension"].transform(inp_hyp)
        results["Hypertension"] = round(models["Hypertension"].predict_proba(scaled_hyp)[0][1] * 100, 2)
    else:
        results["Hypertension"] = "Dataset Not Available"

    # Stroke Prediction
    if "Stroke" in models:
        avg_glucose_est = 100.0  # Population average baseline fallback
        inp_str = np.array([[approx_age, high_bp, heart_hist, avg_glucose_est, bmi, sex]])
        scaled_str = scalers["Stroke"].transform(inp_str)
        results["Stroke"] = round(models["Stroke"].predict_proba(scaled_str)[0][1] * 100, 2)
    else:
        results["Stroke"] = "Dataset Not Available"

    # Heart Failure Prediction
    if "Heart Failure" in models:
        num_features = scalers["Heart Failure"].mean_.shape[0]
        inp_hrt = np.tile(scalers["Heart Failure"].mean_, (1, 1))
        
        # Populate known inputs safely into standard Heart Disease features if present
        inp_hrt[0, 0] = approx_age
        if num_features > 1: inp_hrt[0, 1] = sex
        if num_features > 3: inp_hrt[0, 3] = 130.0 if high_bp == 1 else 115.0 # Resting BP estimate
        if num_features > 4: inp_hrt[0, 4] = 240.0 if high_chol == 1 else 190.0 # Chol estimate
        if num_features > 7: inp_hrt[0, 7] = real_time_hr # Max HR from vitals
        
        scaled_hrt = scalers["Heart Failure"].transform(inp_hrt)
        results["Heart Failure"] = round(models["Heart Failure"].predict_proba(scaled_hrt)[0][1] * 100, 2)
    else:
        results["Heart Failure"] = "Dataset Not Available"

    # Real-time physiological alerts
    alerts = []
    if real_time_hr > 100 or real_time_hr < 60:
        alerts.append(f"⚠️ **Tachycardia/Bradycardia Warning**: Heart Rate is {real_time_hr} bpm.")
    if real_time_spo2 < 95:
        alerts.append(f"⚠️ **Hypoxia Alert**: SpO2 level is critically low at {real_time_spo2}%")
    if not alerts:
        alerts.append("✅ Real-time physiological vitals are within normal stable limits.")
        
    # Recommendations
    recommendations = []
    if isinstance(results["Diabetes"], (int, float)) and results["Diabetes"] > 50:
        recommendations.append("• **Diabetes Protocol**: Recommend HbA1c lab verification and dietary consultation.")
    if isinstance(results["Hypertension"], (int, float)) and (results["Hypertension"] > 50 or high_bp == 1):
        recommendations.append("• **Hypertension Protocol**: Initiate 24-hour ambulatory blood pressure monitoring.")
    if (isinstance(results["Heart Failure"], (int, float)) and results["Heart Failure"] > 40) or \
       (isinstance(results["Stroke"], (int, float)) and results["Stroke"] > 40):
        recommendations.append("• **Cardiovascular Protocol**: Urgent cardiology referral and ECG/Echo evaluation advised.")
    if not recommendations:
        recommendations.append("• Routine annual screening recommended. No acute intervention required.")

    # Helper function to safely format output string values
    def fmt_val(val):
        return f"{val}%" if isinstance(val, (int, float)) else str(val)

    # Format report
    output_report = textwrap.dedent(f"""
    ### 👤 Patient: {name}
    
    ### 📊 Disease Probability Estimates (%)
    * **Diabetes Risk**: \t {fmt_val(results['Diabetes'])}
    * **Hypertension Risk**: \t {fmt_val(results['Hypertension'])}
    * **Heart Failure Risk**: \t {fmt_val(results['Heart Failure'])}
    * **Stroke Risk**: \t {fmt_val(results['Stroke'])}

    ### 🩺 Real-Time Physiological Status
    {'\n'.join(alerts)}

    ### 💡 Clinical Decision Support & Recommendations
    {'\n'.join(recommendations)}
    """)
    return output_report

# 4. INTERACTIVE WEB USER INTERFACE (using Gradio)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan", neutral_hue="slate")) as interface:
    gr.Markdown("### 🏥 AI-Powered Disease Prediction & Decision Support System")
    gr.Markdown("Combine patient historical data with real-time physiological vitals for early warnings and clinical guidance.")
    gr.Markdown("---")
    gr.Markdown("Please set the following inputs for the patient below:")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🧬 Patient Medical History")
            name = gr.Textbox(label="Patient Name", placeholder="Enter patient's name", max_length=50)
            age = gr.Dropdown(choices=[("18-24", 1),("25-29", 2),("30-34", 3),("35-39", 4),("40-44", 5),("45-49", 6),("50-54", 7),("55-59", 8),("60-64", 9),("65-69", 10),("70-74", 11),("75-79", 12),("80 or older", 13)], value=1, label="Age Group")
            high_bp = gr.Dropdown([("No", 0), ("Yes", 1)], label="High Blood Pressure (No/Yes)", value=0)
            high_chol = gr.Dropdown([("No", 0), ("Yes", 1)], label="High Cholesterol (No/Yes)", value=0)
            bmi = gr.Slider(10, 50, value=15, label="Body Mass Index (BMI)")
            smoker = gr.Dropdown([("No", 0), ("Yes", 1)], label="Smoker (At least 5 packs in lifetime) (No/Yes)", value=0)
            stroke_hist = gr.Dropdown([("No", 0), ("Yes", 1)], label="History of Stroke (No/Yes)", value=0)
            heart_hist = gr.Dropdown([("No", 0), ("Yes", 1)], label="History of Heart Disease/Attack (No/Yes)", value=0)
            phys_activity = gr.Dropdown([("No", 0), ("Yes", 1)], label="Physical Activity in Past 30 Days (No/Yes)", value=0)
            diff_walk = gr.Dropdown([("No", 0), ("Yes", 1)], label="Difficulty Walking / Climbing Stairs (No/Yes)", value=0)
            sex = gr.Dropdown([("Female", 0), ("Male", 1)], label="Sex (Female/ Male)", value=0)
            education = gr.Dropdown([("Primary", 0), ("JHS", 1), ("SHS", 2), ("Tertiary", 3)], label="Education Level", value=0)
            income = gr.Slider(1, 8, step=1, value=1, label="Income Scale (GHS) (1:<10k | 2:10–15k | 3:15–20k | 4:20–25k)", info="5:25–35k | 6:35–50k | 7:50–75k | 8:>75k")
            gen_hlth = gr.Slider(1, 5, step=1, value=1, label="General Health Scale (1 = Excellent, 5 = Poor)")
            ment_hlth = gr.Slider(0, 30, value=1, label="Days of Poor Mental Health (Past 30 days)", step=1)
            phys_hlth = gr.Slider(0, 30, value=1, label="Days of Poor Physical Health (Past 30 days)", step=1)
            
        with gr.Column():
            gr.Markdown("### 📡 Real-Time Physiological Stream Vitals")
            real_time_hr = gr.Slider(40, 180, value=40, label="Heart Rate (bpm)", step=1)
            real_time_spo2 = gr.Slider(70, 100, value=70, label="Blood Oxygen (SpO2 %)")
            
            submit_btn = gr.Button("Run Analysis", variant="primary", scale=1)
            
            gr.Markdown("### 📋 Clinical Assessment & Recommendations")
            output_display = gr.Markdown()

    submit_btn.click(
        fn=predict_clinical_risks,
        inputs=[high_bp, high_chol, bmi, smoker, stroke_hist, heart_hist, 
                phys_activity, gen_hlth, ment_hlth, phys_hlth, diff_walk, 
                sex, age, education, income, real_time_hr, real_time_spo2, name],
        outputs=output_display
    )

interface.launch(inline=True, share=True)