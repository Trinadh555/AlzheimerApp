import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading dataset...")
df = pd.read_csv('alzheimers_disease_data.csv')

# Base features definition
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

X = df[base_features].copy()
y = df['Diagnosis']

print("Generating engineered features...")
X['Cholesterol_Ratio'] = X['CholesterolLDL'] / (X['CholesterolHDL'] + 1e-5)

health_cols = ['CardiovascularDisease', 'Diabetes', 'Hypertension', 'Depression', 'HeadInjury']
X['Comorbidity_Score'] = X[health_cols].sum(axis=1)

symptom_cols = ['Confusion', 'Disorientation', 'MemoryComplaints', 'Forgetfulness', 'PersonalityChanges']
X['Symptom_Severity_Index'] = X[symptom_cols].sum(axis=1)

def get_bmi_category(bmi):
    if bmi < 18.5: return 'Underweight'
    elif 18.5 <= bmi < 25: return 'Normal'
    elif 25 <= bmi < 30: return 'Overweight'
    else: return 'Obese'

X['BMI_Category'] = X['BMI'].apply(get_bmi_category)

def get_bp_category(row):
    sys = row['SystolicBP']
    dia = row['DiastolicBP']
    if sys < 120 and dia < 80: return 'Normal'
    elif (120 <= sys <= 129) and dia < 80: return 'Elevated'
    elif (130 <= sys <= 139) or (80 <= dia <= 89): return 'Stage1_Hypertension'
    else: return 'Stage2_Hypertension'

X['BP_Category'] = X.apply(get_bp_category, axis=1)

# One-hot encoding for categorical columns
X = pd.get_dummies(X, columns=['BMI_Category', 'BP_Category'], drop_first=False)

# Train-Test Split (80% train, 20% test to check overfitting)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Regularized XGBoost Model...")
model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=3,           # Overfitting control cheyadaniki tree depth taggichamu
    learning_rate=0.05,    # Learning rate control
    subsample=0.8,         # Row sampling
    colsample_bytree=0.8,  # Column sampling
    min_child_weight=3,    # Leaf node minimum samples
    random_state=42
)
model.fit(X_train, y_train)

# Accuracy evaluations
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the regularized model
joblib.dump(model, 'xgb_model.pkl')
print("SUCCESS: xgb_model.pkl created successfully with overfitting controls!")