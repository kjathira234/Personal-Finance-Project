import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    label_binarize
)

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("budget_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# =========================================================
# 2. CREATE ENCODERS
# =========================================================

gender_encoder = LabelEncoder()
education_encoder = LabelEncoder()
employment_encoder = LabelEncoder()
loan_encoder = LabelEncoder()
target_encoder = LabelEncoder()


# =========================================================
# 3. ENCODE CATEGORICAL FEATURES
# =========================================================

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


# =========================================================
# 4. DISPLAY ENCODER CLASSES
# =========================================================

print("\n=========================================")
print("ENCODER CLASSES")
print("=========================================")

print("Gender:")
print(gender_encoder.classes_)

print("\nEducation:")
print(education_encoder.classes_)

print("\nEmployment:")
print(employment_encoder.classes_)

print("\nLoan:")
print(loan_encoder.classes_)

print("\nFinancial Health:")
print(target_encoder.classes_)


# =========================================================
# 5. SELECT FEATURES
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

X = df[selected_features]

y = df["financial_health_status"]


# =========================================================
# 6. CLASS DISTRIBUTION
# =========================================================

print("\n=========================================")
print("CLASS DISTRIBUTION")
print("=========================================")

print(y.value_counts())

print("\nClass Distribution Percentage:")

print(
    y.value_counts(normalize=True) * 100
)


# =========================================================
# 7. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 8. FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# =========================================================
# 9. CREATE SVM MODEL
# =========================================================

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)


# =========================================================
# 10. TRAIN SVM MODEL
# =========================================================

print("\n=========================================")
print("TRAINING SVM MODEL")
print("=========================================")

svm_model.fit(
    X_train_scaled,
    y_train
)

print("SVM training completed successfully!")


# =========================================================
# 11. SVM PREDICTION
# =========================================================

svm_prediction = svm_model.predict(
    X_test_scaled
)


# =========================================================
# 12. SVM ACCURACY
# =========================================================

svm_accuracy = accuracy_score(
    y_test,
    svm_prediction
)

print("\n=========================================")
print("SVM ACCURACY")
print("=========================================")

print(
    "SVM Accuracy =",
    svm_accuracy
)

print(
    "SVM Accuracy (%) =",
    svm_accuracy * 100
)


# =========================================================
# 13. SVM CONFUSION MATRIX
# =========================================================

svm_cm = confusion_matrix(
    y_test,
    svm_prediction
)

print("\n=========================================")
print("SVM CONFUSION MATRIX")
print("=========================================")

print(svm_cm)


# =========================================================
# 14. SVM CLASSIFICATION REPORT
# =========================================================

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


# =========================================================
# 15. SAVE SVM CONFUSION MATRIX GRAPH
# =========================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=svm_cm,
    display_labels=target_encoder.classes_
)

disp.plot(
    ax=ax
)

ax.set_title(
    "SVM Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSVM confusion matrix graph saved successfully!"
)

print(
    "File: svm_confusion_matrix.png"
)


# =========================================================
# 16. SVM ROC CURVE
# =========================================================

svm_scores = svm_model.decision_function(
    X_test_scaled
)

y_test_binary = label_binarize(
    y_test,
    classes=[0, 1, 2]
)

plt.figure(
    figsize=(8, 6)
)

for i, class_name in enumerate(
    target_encoder.classes_
):

    fpr, tpr, thresholds = roc_curve(
        y_test_binary[:, i],
        svm_scores[:, i]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} (AUC = {roc_auc:.3f})"
    )


# Random classifier line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "SVM ROC Curve"
)

plt.legend(
    loc="lower right"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    "svm_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSVM ROC curve saved successfully!"
)

print(
    "File: svm_roc_curve.png"
)


# =========================================================
# 17. SAVE SVM MODEL
# =========================================================

joblib.dump(
    svm_model,
    "svm_model.pkl"
)


# =========================================================
# 18. SAVE SCALER
# =========================================================

joblib.dump(
    scaler,
    "svm_scaler.pkl"
)


# =========================================================
# 19. SAVE ENCODERS
# =========================================================

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


# =========================================================
# 20. SAVE FEATURE ORDER
# =========================================================

joblib.dump(
    selected_features,
    "selected_features.pkl"
)


# =========================================================
# 21. SAVE SVM RESULTS
# =========================================================

results = {

    "svm_accuracy": svm_accuracy,

    "svm_cm": svm_cm,

    "y_test": y_test,

    "svm_prediction": svm_prediction

}

joblib.dump(
    results,
    "svm_results.pkl"
)


# =========================================================
# 22. FINAL OUTPUT
# =========================================================

print("\n=========================================")
print("SVM FILES SAVED SUCCESSFULLY")
print("=========================================")

print("1. svm_model.pkl")
print("2. svm_scaler.pkl")
print("3. gender_encoder.pkl")
print("4. education_encoder.pkl")
print("5. employment_encoder.pkl")
print("6. loan_encoder.pkl")
print("7. target_encoder.pkl")
print("8. selected_features.pkl")
print("9. svm_results.pkl")
print("10. svm_confusion_matrix.png")
print("11. svm_roc_curve.png")

print("\n=========================================")
print("SVM TRAINING AND EVALUATION COMPLETED!")
print("=========================================")