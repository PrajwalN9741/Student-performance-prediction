# 📊 Student Performance Prediction – Flask Web App

This project is a Machine Learning + Flask Web Application that predicts whether a student will Pass or Fail based on academic, social, and personal features.

It includes:
- Multi-algorithm model training (Logistic Regression, SVM, Random Forest, XGBoost)
- Automatic best-model selection
- CSV/Excel upload support
- Visualization dashboards
- Feature Importance + SHAP explanations
- Downloadable predictions
- Clean UI using Flask templates

------------------------------------------------------------
Project Structure
------------------------------------------------------------
.
├── app.py
├── train_model.py
├── student_data.csv
├── model.pkl
├── algo_metrics.json
├── /uploads
├── /outputs
├── /static/charts
└── /templates/upload.html

------------------------------------------------------------
Installation
------------------------------------------------------------

# 1. Create a virtual environment (optional)
python -m venv venv
venv\Scripts\activate             # Windows
# source venv/bin/activate       # Linux/Mac

# 2. Install dependencies
pip install flask pandas numpy scikit-learn matplotlib joblib xgboost shap

------------------------------------------------------------
Training the Model
------------------------------------------------------------

# Run the training script
python train_model.py

train_model.py will:
- Load student_data.csv
- Preprocess features (OneHotEncode + numeric passthrough)
- Train 5 ML models:
    * Logistic Regression
    * SVM (Linear)
    * SVM (RBF)
    * Random Forest
    * XGBoost
- Evaluate each model (Accuracy, Precision, Recall, F1)
- Choose the BEST model
- Save:
    * model.pkl (Pipeline)
    * algo_metrics.json
- Generate charts inside /static/charts:
    * Confusion Matrix (raw)
    * Confusion Matrix (normalized)
    * Algorithm Accuracy Comparison
    * Feature Importance
    * SHAP Summary Plot

------------------------------------------------------------
Running the Web App
------------------------------------------------------------

python app.py

# Open in browser:
http://127.0.0.1:5000/

------------------------------------------------------------
Application Features
------------------------------------------------------------

1) Default Dataset View
- Loads student_data.csv
- Shows predictions for default data
- Shows:
    * Table preview
    * Downloadable CSV
    * All charts
    * Insights summary
    * Training dataset preview
    * Best algorithm details

2) Upload Your Own Data
- Upload CSV or Excel (.csv, .xlsx, .xls)
- App validates required columns
- Generates:
    * Predicted (yes/no)
    * Probability(%)
    * Downloadable output file
    * Charts
    * Insights

------------------------------------------------------------
Charts & Insights (auto-generated)
------------------------------------------------------------

predicted_counts.png
    - Pass/Fail output distribution

studytime_by_pred.png
    - Avg studytime comparison between pass vs fail

absences_hist.png
    - Absence count distribution

failures_by_pred.png
    - Avg number of past failures

corr_heatmap.png
    - Feature correlations with prediction

confusion_matrix_raw.png
    - Classification counts

confusion_matrix_norm.png
    - Normalized confusion matrix

algo_comparison.png
    - Accuracy comparison between all ML models

feature_importance.png
    - Most important features (RF/XGB only)

shap_summary.png
    - SHAP explainability chart

------------------------------------------------------------
Notes
------------------------------------------------------------
- Ensure your uploaded dataset columns match training columns.
- To retrain with new data → update student_data.csv and rerun:
      python train_model.py
- Best model is automatically selected based on accuracy.

------------------------------------------------------------
Author
------------------------------------------------------------
Machine Learning + Flask Mini Project
