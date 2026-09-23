# =========================================================
# AI-BASED PERSONAL FINANCE HEALTH COACH
# SVM MODEL TRAINING
# =========================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# =========================================================
# 1. DATASET PATH
# =========================================================

DATASET_PATH = r"C:\Users\Athira KJ\OneDrive\Desktop\personal finance\budget_dataset.csv"


# =========================================================
# 2. OUTPUT DIRECTORY
# =========================================================

# Save model files in the same folder as svm.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


print("\n")
print("==============================================")
print("AI-BASED PERSONAL FINANCE HEALTH COACH")
print("SVM MODEL TRAINING")
print("==============================================")


# =========================================================
# 3. LOAD DATASET
# =========================================================

print("\nLoading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =========================================================
# 4. BASIC DATASET INFORMATION
# =========================================================

print("\n")
print("==============================================")
print("DATASET INFORMATION")
print("==============================================")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# =========================================================
# 5. REMOVE DUPLICATES IF ANY
# =========================================================

if df.duplicated().sum() > 0:

    print("\nRemoving duplicate rows...")

    df = df.drop_duplicates().reset_index(drop=True)

    print("New dataset shape:", df.shape)


# =========================================================
# 6. CHECK TARGET COLUMN
# =========================================================

target_column = "financial_health_status"

if target_column not in df.columns:

    raise ValueError(
        f"Target column '{target_column}' was not found."
    )


# =========================================================
# 7. CLASS DISTRIBUTION
# =========================================================

print("\n")
print("==============================================")
print("CLASS DISTRIBUTION")
print("==============================================")

class_counts = df[target_column].value_counts()

print(class_counts)

print("\nClass percentages:")

class_percentages = (
    df[target_column]
    .value_counts(normalize=True)
    .mul(100)
    .round(3)
)

print(class_percentages)


# =========================================================
# 8. SELECT EXACTLY 14 FEATURES
# =========================================================

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


print("\n")
print("==============================================")
print("SELECTED FEATURES")
print("==============================================")


for number, feature in enumerate(
    selected_features,
    start=1
):

    print(
        f"{number}. {feature}"
    )


print(
    "\nTotal input features:",
    len(selected_features)
)


# =========================================================
# 9. VERIFY ALL FEATURES EXIST
# =========================================================

missing_features = [

    feature

    for feature in selected_features

    if feature not in df.columns

]


if missing_features:

    print("\nERROR!")

    print(
        "The following features are missing:"
    )

    for feature in missing_features:

        print(
            "-",
            feature
        )

    raise ValueError(
        "Required feature columns are missing from dataset."
    )


# =========================================================
# 10. CREATE X AND y
# =========================================================

X = df[selected_features].copy()

y = df[target_column].copy()


# =========================================================
# 11. CREATE LABEL ENCODERS
# =========================================================

print("\n")
print("==============================================")
print("CREATING LABEL ENCODERS")
print("==============================================")


gender_encoder = LabelEncoder()

education_encoder = LabelEncoder()

employment_encoder = LabelEncoder()

loan_encoder = LabelEncoder()

target_encoder = LabelEncoder()


# =========================================================
# 12. ENCODE GENDER
# =========================================================

X["gender"] = gender_encoder.fit_transform(
    X["gender"].astype(str)
)


# =========================================================
# 13. ENCODE EDUCATION
# =========================================================

X["education_level"] = education_encoder.fit_transform(
    X["education_level"].astype(str)
)


# =========================================================
# 14. ENCODE EMPLOYMENT
# =========================================================

X["employment_status"] = employment_encoder.fit_transform(
    X["employment_status"].astype(str)
)


# =========================================================
# 15. ENCODE LOAN
# =========================================================

X["has_loan"] = loan_encoder.fit_transform(
    X["has_loan"].astype(str)
)


# =========================================================
# 16. ENCODE TARGET
# =========================================================

y = target_encoder.fit_transform(
    y.astype(str)
)


# =========================================================
# 17. DISPLAY ENCODING
# =========================================================

print("\nGender encoding:")

for value, number in zip(
    gender_encoder.classes_,
    range(len(gender_encoder.classes_))
):

    print(
        f"{value} = {number}"
    )


print("\nEducation encoding:")

for value, number in zip(
    education_encoder.classes_,
    range(len(education_encoder.classes_))
):

    print(
        f"{value} = {number}"
    )


print("\nEmployment encoding:")

for value, number in zip(
    employment_encoder.classes_,
    range(len(employment_encoder.classes_))
):

    print(
        f"{value} = {number}"
    )


print("\nLoan encoding:")

for value, number in zip(
    loan_encoder.classes_,
    range(len(loan_encoder.classes_))
):

    print(
        f"{value} = {number}"
    )


print("\nTarget encoding:")

for value, number in zip(
    target_encoder.classes_,
    range(len(target_encoder.classes_))
):

    print(
        f"{value} = {number}"
    )


# =========================================================
# 18. CONVERT ALL FEATURES TO NUMERIC
# =========================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# =========================================================
# 19. HANDLE MISSING VALUES
# =========================================================

if X.isnull().sum().sum() > 0:

    print("\nMissing numeric values detected.")

    print(
        X.isnull().sum()
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    print(
        "Missing values handled."
    )


# =========================================================
# 20. TRAIN / TEST SPLIT
# =========================================================

print("\n")
print("==============================================")
print("TRAIN / TEST SPLIT")
print("==============================================")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records :",
    len(X_test)
)


# =========================================================
# 21. STANDARD SCALER
# =========================================================

print("\n")
print("==============================================")
print("FEATURE SCALING")
print("==============================================")


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


print(
    "StandardScaler applied successfully!"
)


# =========================================================
# 22. CREATE RBF SVM
# =========================================================

print("\n")
print("==============================================")
print("CREATING SVM MODEL")
print("==============================================")


# Lower regularization:
# C = 0.1
#
# This makes the SVM less strict about fitting
# every training sample and can reduce overfitting.

svm_model = SVC(

    kernel="rbf",

    C=0.1,

    gamma="scale",

    class_weight=None,

    random_state=42

)


print(
    "Algorithm    : Support Vector Machine"
)

print(
    "Kernel       : RBF"
)

print(
    "C            : 0.1"
)

print(
    "Gamma        : scale"
)

print(
    "Class Weight : None"
)

print(
    "Probability  : Not enabled"
)


# =========================================================
# 23. TRAIN SVM
# =========================================================

print("\n")
print("==============================================")
print("TRAINING SVM")
print("==============================================")


svm_model.fit(

    X_train_scaled,

    y_train

)


print(
    "\nSVM training completed successfully!"
)


# =========================================================
# 24. MAKE TEST PREDICTIONS
# =========================================================

print("\n")
print("==============================================")
print("MAKING TEST PREDICTIONS")
print("==============================================")


svm_prediction = svm_model.predict(

    X_test_scaled

)


print(
    "Prediction completed!"
)


# =========================================================
# 25. ACCURACY
# =========================================================

accuracy = accuracy_score(

    y_test,

    svm_prediction

)


print("\n")
print("==============================================")
print("SVM ACCURACY")
print("==============================================")


print(
    "SVM Accuracy:",
    accuracy
)


print(
    "SVM Accuracy (%):",
    round(
        accuracy * 100,
        2
    )
)


# =========================================================
# 26. MACRO F1 SCORE
# =========================================================

macro_f1 = f1_score(

    y_test,

    svm_prediction,

    average="macro"

)


print("\n")
print("==============================================")
print("MACRO F1 SCORE")
print("==============================================")


print(
    "Macro F1:",
    round(
        macro_f1,
        4
    )
)


print(
    "Macro F1 (%):",
    round(
        macro_f1 * 100,
        2
    )
)


# =========================================================
# 27. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(

    y_test,

    svm_prediction

)


print("\n")
print("==============================================")
print("CONFUSION MATRIX")
print("==============================================")


print(cm)


# =========================================================
# 28. CLASSIFICATION REPORT
# =========================================================

print("\n")
print("==============================================")
print("SVM CLASSIFICATION REPORT")
print("==============================================")


classification_report_text = classification_report(

    y_test,

    svm_prediction,

    target_names=target_encoder.classes_,

    zero_division=0

)


print(
    classification_report_text
)


# =========================================================
# 29. CLASS-WISE PERFORMANCE
# =========================================================

report = classification_report(

    y_test,

    svm_prediction,

    target_names=target_encoder.classes_,

    zero_division=0,

    output_dict=True

)


print("\n")
print("==============================================")
print("CLASS-WISE PERFORMANCE")
print("==============================================")


for class_name in target_encoder.classes_:

    print(

        f"{class_name}: "

        f"Precision = "

        f"{report[class_name]['precision']:.3f}, "

        f"Recall = "

        f"{report[class_name]['recall']:.3f}, "

        f"F1 = "

        f"{report[class_name]['f1-score']:.3f}"

    )


# =========================================================
# 30. PREDICTED CLASS DISTRIBUTION
# =========================================================

predicted_class_names = target_encoder.inverse_transform(

    svm_prediction

)


predicted_distribution = pd.Series(

    predicted_class_names

).value_counts()


print("\n")
print("==============================================")
print("PREDICTED CLASS DISTRIBUTION")
print("==============================================")


print(
    predicted_distribution
)


print(
    "\nPredicted percentages:"
)


predicted_percentages = (

    pd.Series(predicted_class_names)

    .value_counts(normalize=True)

    .mul(100)

    .round(3)

)


print(
    predicted_percentages
)


# =========================================================
# 31. COMPARE ACTUAL VS PREDICTED CLASS COUNTS
# =========================================================

actual_class_names = target_encoder.inverse_transform(

    y_test

)


actual_distribution = pd.Series(

    actual_class_names

).value_counts()


comparison = pd.DataFrame({

    "Actual": actual_distribution,

    "Predicted": predicted_distribution

}).fillna(0).astype(int)


print("\n")
print("==============================================")
print("ACTUAL VS PREDICTED DISTRIBUTION")
print("==============================================")


print(
    comparison
)


# =========================================================
# 32. SAVE MODEL
# =========================================================

print("\n")
print("==============================================")
print("SAVING MODEL")
print("==============================================")


model_path = os.path.join(

    BASE_DIR,

    "svm_model.pkl"

)


joblib.dump(

    svm_model,

    model_path

)


print(
    "svm_model.pkl saved successfully!"
)


# =========================================================
# 33. SAVE SCALER
# =========================================================

scaler_path = os.path.join(

    BASE_DIR,

    "svm_scaler.pkl"

)


joblib.dump(

    scaler,

    scaler_path

)


print(
    "svm_scaler.pkl saved successfully!"
)


# =========================================================
# 34. SAVE ALL ENCODERS TOGETHER
# =========================================================

svm_encoders = {

    "gender": gender_encoder,

    "education_level": education_encoder,

    "employment_status": employment_encoder,

    "has_loan": loan_encoder,

    "target": target_encoder

}


encoders_path = os.path.join(

    BASE_DIR,

    "svm_encoders.pkl"

)


joblib.dump(

    svm_encoders,

    encoders_path

)


print(
    "svm_encoders.pkl saved successfully!"
)


# =========================================================
# 35. SAVE SELECTED FEATURES
# =========================================================

features_path = os.path.join(

    BASE_DIR,

    "selected_features.pkl"

)


joblib.dump(

    selected_features,

    features_path

)


print(
    "selected_features.pkl saved successfully!"
)


# =========================================================
# 36. SAVE RESULTS
# =========================================================

svm_results = {

    "accuracy": accuracy,

    "accuracy_percentage": accuracy * 100,

    "macro_f1": macro_f1,

    "macro_f1_percentage": macro_f1 * 100,

    "classification_report": report,

    "confusion_matrix": cm,

    "selected_features": selected_features,

    "target_classes": list(

        target_encoder.classes_

    ),

    "model_parameters": {

        "algorithm": "SVM",

        "kernel": "rbf",

        "C": 0.1,

        "gamma": "scale",

        "class_weight": None

    }

}


results_path = os.path.join(

    BASE_DIR,

    "svm_results.pkl"

)


joblib.dump(

    svm_results,

    results_path

)


print(
    "svm_results.pkl saved successfully!"
)


# =========================================================
# 37. FINAL SUMMARY
# =========================================================

print("\n")
print("==============================================")
print("TRAINING COMPLETED")
print("==============================================")


print(

    f"\nFinal SVM Accuracy: "

    f"{accuracy * 100:.2f}%"

)


print(

    f"Final Macro F1: "

    f"{macro_f1 * 100:.2f}%"

)


print(

    "\nTotal input features:",

    len(selected_features)

)


print(
    "\nFeatures used:"
)


for feature in selected_features:

    print(
        " -",
        feature
    )


print(
    "\nGenerated files:"
)

print(
    " - svm_model.pkl"
)

print(
    " - svm_scaler.pkl"
)

print(
    " - svm_encoders.pkl"
)

print(
    " - selected_features.pkl"
)

print(
    " - svm_results.pkl"
)


print("\n")
print("==============================================")
print("SVM TRAINING FINISHED")
print("==============================================")