import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Alzheimer's Disease Prediction App",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Alzheimer's Disease Prediction System")
st.markdown("This application uses your trained **XGBoost Machine Learning Model** with automated feature engineering and batch dataset evaluation.")

# Sidebar Navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Choose Mode", ["Single Patient Prediction", "Batch CSV Prediction"])

# Load trained model safely
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgb_model.pkl')
    except Exception as e:
        return None

model = load_model()

# Helper function for automated feature processing
def process_features(df):
    base_features = [
        'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking', 
        'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 
        'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 
        'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
        'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 
        'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL', 
        'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 
        'Forgetfulness'
    ]
    
    processed_df = pd.DataFrame()
    for col in base_features:
        if col in df.columns:
            processed_df[col] = df[col]
        else:
            processed_df[col] = 0

    # Automated Feature Engineering
    processed_df['Cholesterol_Ratio'] = processed_df['CholesterolLDL'] / (processed_df['CholesterolHDL'] + 1e-5)
    
    health_cols = ['CardiovascularDisease', 'Diabetes', 'Hypertension', 'Depression', 'HeadInjury']
    processed_df['Comorbidity_Score'] = processed_df[health_cols].sum(axis=1)
    
    symptom_cols = ['Confusion', 'Disorientation', 'MemoryComplaints', 'Forgetfulness', 'PersonalityChanges']
    processed_df['Symptom_Severity_Index'] = processed_df[symptom_cols].sum(axis=1)

    processed_df['BMI_Category'] = processed_df['BMI'].apply(
        lambda x: 'Underweight' if x < 18.5 else ('Normal' if 18.5 <= x < 25 else ('Overweight' if 25 <= x < 30 else 'Obese'))
    )
    
    def batch_bp(r):
        sys, dia = r['SystolicBP'], r['DiastolicBP']
        if sys < 120 and dia < 80: return 'Normal'
        elif (120 <= sys <= 129) and dia < 80: return 'Elevated'
        elif (130 <= sys <= 139) or (80 <= dia <= 89): return 'Stage1_Hypertension'
        else: return 'Stage2_Hypertension'
    
    processed_df['BP_Category'] = processed_df.apply(batch_bp, axis=1)
    processed_df = pd.get_dummies(processed_df, columns=['BMI_Category', 'BP_Category'], drop_first=False)

    if model is not None:
        expected_features = model.get_booster().feature_names
        for col in expected_features:
            if col not in processed_df.columns:
                processed_df[col] = 0
        processed_df = processed_df[expected_features]

    return processed_df

if app_mode == "Single Patient Prediction":
    st.subheader("Patient Clinical & Demographic Evaluation Form")
    st.info("Fill out the details across the tabs below. All required fields including Cholesterol levels are included.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Demographics & Lifestyle", 
        "🩺 Vitals & Lab Results", 
        "🧩 Cognitive & Symptoms", 
        "⚙️ Engineered Features Preview"
    ])

    with tab1:
        st.subheader("Demographics & Lifestyle")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 50, 90, 75, key="single_age")
            gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female", key="single_gender")
            ethnicity = st.selectbox("Ethnicity", options=[0, 1, 2, 3], format_func=lambda x: ["Caucasian", "African American", "Asian", "Other"][x], key="single_eth")
        with col2:
            education_level = st.selectbox("Education Level", options=[0, 1, 2, 3], format_func=lambda x: ["None", "High School", "Bachelor's", "Graduate"][x], key="single_edu")
            bmi = st.number_input("BMI", 15.0, 40.0, 27.5, key="single_bmi")
            smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_smoke")
        with col3:
            alcohol = st.number_input("Alcohol Consumption", 0.0, 20.0, 10.0, key="single_alc")
            physical_activity = st.number_input("Physical Activity", 0.0, 10.0, 5.0, key="single_pa")
            diet_quality = st.number_input("Diet Quality", 0.0, 10.0, 5.0, key="single_diet")
            sleep_quality = st.number_input("Sleep Quality", 4.0, 10.0, 7.0, key="single_sleep")

    with tab2:
        st.subheader("Medical History, Vitals & Complete Cholesterol Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            family_history = st.selectbox("Family History of Alzheimer's", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_fh")
            cvd = st.selectbox("Cardiovascular Disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_cvd")
            diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_diab")
            depression = st.selectbox("Depression", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_dep")
        with col2:
            head_injury = st.selectbox("History of Head Injury", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_hi")
            hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_ht")
            systolic_bp = st.number_input("Systolic BP", 90, 180, 134, key="single_sys")
            diastolic_bp = st.number_input("Diastolic BP", 60, 120, 89, key="single_dia")
        with col3:
            chol_total = st.number_input("Cholesterol Total", 150.0, 300.0, 225.0, key="single_ct")
            chol_ldl = st.number_input("Cholesterol LDL", 50.0, 200.0, 124.0, key="single_ldl")
            chol_hdl = st.number_input("Cholesterol HDL", 20.0, 100.0, 59.0, key="single_hdl")
            chol_trig = st.number_input("Cholesterol Triglycerides", 50.0, 400.0, 229.0, key="single_trig")

    with tab3:
        st.subheader("Cognitive & Behavioral Assessments")
        col1, col2 = st.columns(2)
        with col1:
            mmse = st.number_input("MMSE Score", 0.0, 30.0, 14.7, key="single_mmse")
            functional_assessment = st.number_input("Functional Assessment", 0.0, 10.0, 5.0, key="single_fa")
            memory_complaints = st.selectbox("Memory Complaints", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_mc")
            behavioral_problems = st.selectbox("Behavioral Problems", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_bp")
            adl = st.number_input("ADL Score", 0.0, 10.0, 4.9, key="single_adl")
        with col2:
            confusion = st.selectbox("Confusion Episodes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_conf")
            disorientation = st.selectbox("Disorientation", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_dis")
            personality_changes = st.selectbox("Personality Changes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_pc")
            difficulty_tasks = st.selectbox("Difficulty Completing Tasks", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_dt")
            forgetfulness = st.selectbox("Forgetfulness", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="single_ff")

    with tab4:
        st.subheader("Advanced Engineered Features Preview")
        col_ratio_val = chol_ldl / (chol_hdl + 1e-5)
        comorbidity_val = int(cvd + diabetes + hypertension + depression + head_injury)
        symptom_val = int(confusion + disorientation + memory_complaints + forgetfulness + personality_changes)
        
        st.metric("Calculated Cholesterol Ratio (LDL / HDL)", f"{col_ratio_val:.2f}")
        st.metric("Comorbidity Score", comorbidity_val)
        st.metric("Symptom Severity Index", symptom_val)

    st.markdown("---")
    if st.button("Run Diagnostic Prediction", type="primary", key="single_predict_btn"):
        input_data = {
            'Age': age, 'Gender': gender, 'Ethnicity': ethnicity, 'EducationLevel': education_level,
            'BMI': bmi, 'Smoking': smoking, 'AlcoholConsumption': alcohol, 'PhysicalActivity': physical_activity,
            'DietQuality': diet_quality, 'SleepQuality': sleep_quality, 'FamilyHistoryAlzheimers': family_history,
            'CardiovascularDisease': cvd, 'Diabetes': diabetes, 'Depression': depression, 'HeadInjury': head_injury,
            'Hypertension': hypertension, 'SystolicBP': systolic_bp, 'DiastolicBP': diastolic_bp,
            'CholesterolTotal': chol_total, 'CholesterolLDL': chol_ldl, 'CholesterolHDL': chol_hdl,
            'CholesterolTriglycerides': chol_trig, 'MMSE': mmse, 'FunctionalAssessment': functional_assessment,
            'MemoryComplaints': memory_complaints, 'BehavioralProblems': behavioral_problems, 'ADL': adl,
            'Confusion': confusion, 'Disorientation': disorientation, 'PersonalityChanges': personality_changes,
            'DifficultyCompletingTasks': difficulty_tasks, 'Forgetfulness': forgetfulness
        }

        input_df = pd.DataFrame([input_data])
        processed_input = process_features(input_df)

        if model is not None:
            try:
                prediction = model.predict(processed_input)[0]
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error("⚠️ **High Risk of Alzheimer's Disease Detected**")
                else:
                    st.success("✅ **Low Risk / Normal**")
            except Exception as ex:
                st.error(f"Prediction Error: {ex}")
        else:
            st.warning("Model file (`xgb_model.pkl`) not found in directory.")

elif app_mode == "Batch CSV Prediction":
    st.subheader("Batch Dataset Evaluation")
    uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"], key="batch_csv_upload")
    
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", test_df.head())
        
        total_rows = len(test_df)
        st.success(f"CSV successfully loaded! Total rows available: **{total_rows}**")
        
        selected_rows = st.slider(
            "Select number of rows for prediction:", 
            min_value=1, 
            max_value=total_rows, 
            value=min(20, total_rows),
            key="row_slider"
        )
        
        test_df_subset = test_df.head(selected_rows).copy()

        if st.button("Predict Batch", key="batch_predict_btn"):
            if model is not None:
                try:
                    final_test_df = process_features(test_df_subset)
                    preds = model.predict(final_test_df)
                    
                    # Clean output dataframe with only PatientID and Diagnosis Result
                    result_df = pd.DataFrame()
                    if 'PatientID' in test_df_subset.columns:
                        result_df['PatientID'] = test_df_subset['PatientID']
                    else:
                        result_df['Row_Index'] = range(len(preds))
                        
                    result_df['Diagnosis_Result'] = ["High Risk" if p == 1 else "Low Risk / Normal" for p in preds]

                    st.success(f"Batch predictions successfully completed for {selected_rows} rows!")
                    st.dataframe(result_df)
                    
                    csv_data = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_data,
                        file_name='alzheimers_predictions.csv',
                        mime='text/csv',
                        key="download_csv_btn"
                    )
                except Exception as ex:
                    st.error(f"Batch Error: {ex}")
            else:
                st.error("Model file not loaded.")