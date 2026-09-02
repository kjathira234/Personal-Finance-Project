import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
from sklearn.svm import SVC

import joblib


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("budget_dataset.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())


# =========================================
# ENCODE CATEGORICAL DATA
# =========================================

gender_encoder = LabelEncoder()
education_encoder = LabelEncoder()
employment_encoder = LabelEncoder()
loan_encoder = LabelEncoder()
target_encoder = LabelEncoder()

df["gender"] = gender_encoder.fit_transform(
    df["gender"]
)

df["education_level"] = education_encoder.fit_transform(
    df["education_level"]
)

df["employment_status"] = employment_encoder.fit_transform(
    df["employment_status"]
)

df["has_loan"] = loan_encoder.fit_transform(
    df["has_loan"]
)

df["financial_health_status"] = target_encoder.fit_transform(
    df["financial_health_status"]
)

print(
    "Financial Health Status Classes:",
    target_encoder.classes_
)

print(df.head())


# =========================================
# SELECT FEATURES
# =========================================

selected_features = [
    "age",
    "gender",
    "education_level",
    "employment_status",
    "monthly_income_usd",
    "monthly_expenses_usd",
    "savings_usd",
    "has_loan",
    "loan_amount_usd",
    "monthly_emi_usd",
    "debt_to_income_ratio",
    "credit_score",
    "savings_to_income_ratio",
    "expense_ratio"
]

X = df[selected_features]

y = df["financial_health_status"]

print("\nClass Distribution:")
print(y.value_counts())

print("\nClass Distribution Percentage:")
print(y.value_counts(normalize=True) * 100)

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# =========================================
# RANDOM FOREST
# MAIN FINANCIAL HEALTH MODEL
# =========================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_prediction = rf_model.predict(
    X_test
)
# Probability prediction for Random Forest ROC curve
rf_probability = rf_model.predict_proba(X_test)


rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)

print("\nRandom Forest Accuracy =", rf_accuracy)

print(
    "Random Forest Accuracy (%) =",
    rf_accuracy * 100
)


# =========================================
# DECISION TREE
# =========================================

dt_model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

dt_prediction = dt_model.predict(
    X_test
)

dt_accuracy = accuracy_score(
    y_test,
    dt_prediction
)

print(
    "\nDecision Tree Accuracy =",
    dt_accuracy
)

print(
    "Decision Tree Accuracy (%) =",
    dt_accuracy * 100
)


# =========================================
# XGBOOST
# =========================================

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss"
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_prediction = xgb_model.predict(
    X_test
)

# Probability prediction for ROC curve
xgb_probability = xgb_model.predict_proba(
    X_test
)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_prediction
)

print(
    "\nXGBoost Accuracy =",
    xgb_accuracy
)

print(
    "XGBoost Accuracy (%) =",
    xgb_accuracy * 100
)


# =========================================
# FEATURE SCALING FOR SVM
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# =========================================
# SUPPORT VECTOR MACHINE
# =========================================

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)

svm_model.fit(
    X_train_scaled,
    y_train
)

svm_prediction = svm_model.predict(
    X_test_scaled
)


svm_accuracy = accuracy_score(
    y_test,
    svm_prediction
)

print(
    "\nSVM Accuracy =",
    svm_accuracy
)

print(
    "SVM Accuracy (%) =",
    svm_accuracy * 100
)


# =========================================
# ALGORITHM COMPARISON
# =========================================

print("\n=========================================")
print("ALGORITHM ACCURACY COMPARISON")
print("=========================================")

print(
    "Random Forest Accuracy =",
    rf_accuracy
)

print(
    "Decision Tree Accuracy =",
    dt_accuracy
)

print(
    "XGBoost Accuracy =",
    xgb_accuracy
)

print(
    "SVM Accuracy =",
    svm_accuracy
)


# =========================================
# RANDOM FOREST CONFUSION MATRIX
# =========================================

cm = confusion_matrix(
    y_test,
    rf_prediction
)

print("\n=========================================")
print("RANDOM FOREST CONFUSION MATRIX")
print("=========================================")

print(cm)


# =========================================
# RANDOM FOREST CLASSIFICATION REPORT
# =========================================

print("\n=========================================")
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=========================================")

print(
    classification_report(
        y_test,
        rf_prediction,
        target_names=target_encoder.classes_,
        zero_division=0
    )
)


# =========================================
# CLASS DISTRIBUTION
# =========================================

print("\n=========================================")
print("FINANCIAL HEALTH STATUS DISTRIBUTION")
print("=========================================")

print(
    df["financial_health_status"].value_counts()
)


# =========================================
# SAVE SVM MODEL
# =========================================

joblib.dump(
    svm_model,
    "finance_model.pkl"
)

# Save the scaler used by SVM
joblib.dump(
    scaler,
    "svm_scaler.pkl"
)

print("\nSVM model saved successfully!")
print("SVM scaler saved successfully!")


# =========================================
# SAVE ENCODERS
# =========================================

joblib.dump(
    gender_encoder,
    "gender_encoder.pkl"
)

joblib.dump(
    education_encoder,
    "education_encoder.pkl"
)

joblib.dump(
    employment_encoder,
    "employment_encoder.pkl"
)

joblib.dump(
    loan_encoder,
    "loan_encoder.pkl"
)

joblib.dump(
    target_encoder,
    "target_encoder.pkl"
)
joblib.dump(svm_model, "finance_model.pkl")



# =========================================
# SAVE GRAPH RESULTS
# =========================================

results = {
    "rf_accuracy": rf_accuracy,
    "dt_accuracy": dt_accuracy,
    "xgb_accuracy": xgb_accuracy,
    "svm_accuracy": svm_accuracy,

    "rf_cm": cm,
    "svm_cm": svm_cm,

    "y_test": y_test,

    "rf_prediction": rf_prediction,
    "xgb_prediction": xgb_prediction,
    "svm_prediction": svm_prediction,

    "rf_probability": rf_probability,
    "xgb_probability": xgb_probability,
}

joblib.dump(
    results,
    "graph_results.pkl"
)

print("Graph results saved successfully!")
print("Saved keys:")
print(results.keys())

joblib.dump(results, "graph_results.pkl")

print("Graph results saved successfully!")
print("Saved keys:")
print(results.keys())



# =========================================
# SVM CONFUSION MATRIX
# =========================================

svm_cm = confusion_matrix(
    y_test,
    svm_prediction
)

print("\n=========================================")
print("SVM CONFUSION MATRIX")
print("=========================================")

print(svm_cm)


# =========================================
# SVM CLASSIFICATION REPORT
# =========================================

print("\n=========================================")
print("SVM CLASSIFICATION REPORT")
print("=========================================")

print(
    classification_report(
        y_test,
        svm_prediction,
        target_names=target_encoder.classes_,
        zero_division=0
    )
)


# =========================================
# SAVE GRAPH RESULTS
# =========================================

results = {
    "rf_accuracy": rf_accuracy,
    "dt_accuracy": dt_accuracy,
    "xgb_accuracy": xgb_accuracy,
    "svm_accuracy": svm_accuracy,

    "rf_cm": cm,
    "svm_cm": svm_cm,

    "y_test": y_test,

    "rf_prediction": rf_prediction,
    "xgb_prediction": xgb_prediction,
    "svm_prediction": svm_prediction,

    "rf_probability": rf_probability,
    "xgb_probability": xgb_probability
}


joblib.dump(
    results,
    "graph_results.pkl"
)

print("Graph results saved successfully!")
print("Saved keys:")
print(results.keys())