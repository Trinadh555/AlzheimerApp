# Predictive Intelligence for Alzheimer's Disease Using XGBoost

A machine learning project for predicting Alzheimer's disease using patient demographic, lifestyle, medical, cognitive, behavioral, and functional information.

The project uses **Exploratory Data Analysis (EDA), Feature Engineering, preprocessing, and XGBoost classification**, with the final model deployed as an interactive **Streamlit web application**.

##  Live Demo

👉 **[Open the Alzheimer's Disease Prediction App](https://alzhemerapp-78m4heigrxhopkknktvs7.streamlit.app/)**

##  problem Statement

Alzheimer's disease is a progressive neurological condition where early identification can support timely assessment.

The objective of this project is to develop a **machine learning classification model** that predicts whether Alzheimer's disease is present based on patient information.

The project uses:

* Demographic information
* Lifestyle factors
* Medical conditions
* Cognitive features
* Behavioral features
* Functional assessment

EDA and feature engineering were performed to identify useful patterns before model training.

##  Dataset

The dataset contains:

* **2,149 records**
* **35 original columns**
* Age range: **60–90**
* Mean BMI: **27.66**
* Target variable: `Diagnosis`

### Target Classes

| Diagnosis | Meaning                |
| --------- | ---------------------- |
| `0`       | No Alzheimer's disease |
| `1`       | Alzheimer's disease    |

### Feature Categories

* **Demographic:** Age, Gender, Ethnicity, EducationLevel
* **Lifestyle:** Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality
* **Medical:** Diabetes, Hypertension, CardiovascularDisease, Depression, HeadInjury, Blood Pressure
* **Cognitive / Functional:** MMSE, FunctionalAssessment, ADL, MemoryComplaints, BehavioralProblems

The dataset contains **1,389 patients without Alzheimer's disease and 760 patients with Alzheimer's disease**.

##  Exploratory Data Analysis & Feature Engineering

EDA showed relatively weak linear correlations for many individual variables, while cognitive and functional features showed more useful predictive patterns. This supported the use of a non-linear tree-based model.

### Feature Engineering

The following features were created:

* **Cholesterol Ratio** — derived from LDL and HDL
* **Comorbidity Score** — based on selected medical conditions
* **Symptom Severity Index** — based on selected cognitive and behavioral symptoms
* **BMI Category**
* **Blood Pressure Category**

Additional preprocessing included:

* Excluding `PatientID` and `DoctorInCharge`
* Converting categorical variables using one-hot encoding
* Preparing the final model input features

The final model input contained **37 features**.

##  Machine Learning Model

Multiple classification approaches were evaluated, including:

* XGBoost
* Gradient Boosting
* AdaBoost
* Logistic Regression
* Naive Bayes
* SVM

**XGBoost** was selected for the final workflow based on the project evaluation.

##  Model Performance

The reported XGBoost test-set results are:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **95.12%** |
| Precision | **95.83%** |
| Recall    | **90.20%** |
| F1-Score  | **92.93%** |
| ROC-AUC   | **0.9522** |

The model comparison also reported XGBoost at **95.4% accuracy**, compared with 95.1% for Tuned Gradient Boosting, 91% for AdaBoost, 82.79% for Logistic Regression, 81.16% for Naive Bayes, and 80.7% for SVM.

##  Hyperparameter Tuning

Hyperparameter tuning was performed to control model complexity and improve generalization.

The project evaluated parameters including:

* `max_depth`
* `learning_rate`
* `n_estimators`
* `subsample`
* `colsample_bytree`
* `min_child_weight`
* `gamma`

The reported results showed:

* Initial training accuracy: **100.00%**
* Initial test accuracy: **95.12%**
* Tuned training accuracy: **96.97%**
* Tuned test accuracy: **95.12%**

The tuning reduced the train-test gap and improved the model's generalization behaviour.

##  Cross-Validation

A **5-Fold Stratified Cross-Validation** approach was used.

The reported mean validation accuracy was:

**94.47%**

Each fold was used once for validation while the remaining four folds were used for training.

##  Deployment

The trained model was saved using **Joblib/PKL** and reused for prediction.

### Deployment Workflow

```text
Patient Input
      ↓
Python Backend
      ↓
XGBoost Model
      ↓
Prediction
```

The application was deployed using **Streamlit** and uses the saved `xgb_model.pkl` model for prediction.

##  Project Structure

```text
Predictive-Intelligence-for-Alzheimer-s-Disease-Using-XGBoost/
│
├── app.py
├── train_and_save.py
├── requirements.txt
├── xgb_model.pkl
├── alzheimers_disease_data.csv
└── README.md
```

##  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* Streamlit
* Machine Learning
* Feature Engineering

##  Run Locally

Clone the repository:

```bash
git clone https://github.com/Trinadh555/Predictive-Intelligence-for-Alzheimer-s-Disease-Using-XGBoost.git
```

Navigate to the project directory:

```bash
cd Predictive-Intelligence-for-Alzheimer-s-Disease-Using-XGBoost
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

##  Future Scope

Future improvements include:

* Validating the model on larger and more representative datasets
* Adding Explainable AI to understand individual predictions
* Exploring additional feature-selection methods
* Further hyperparameter optimization
* Evaluating calibrated prediction probabilities
* Adding additional classification models
* Developing a more user-friendly real-time application

##  Disclaimer

This project is intended for **educational and demonstration purposes only**. It is not a substitute for professional medical diagnosis or clinical decision-making.
