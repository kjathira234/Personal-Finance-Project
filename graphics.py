import pandas as pd
import matplotlib

# Use Tkinter backend
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("budget_dataset.csv")


# =========================================
# LOAD MODEL RESULTS
# =========================================

results = joblib.load("graph_results.pkl")


rf_accuracy = results["rf_accuracy"]
dt_accuracy = results["dt_accuracy"]
xgb_accuracy = results["xgb_accuracy"]
svm_accuracy = results["svm_accuracy"]

cm = results["cm"]

y_test = results["y_test"]

xgb_prediction = results["xgb_prediction"]

xgb_probability = results["xgb_probability"]


# =========================================
# CLASS NAMES
# =========================================

classes = [
    "Average",
    "Good",
    "Poor"
]


# =========================================
# PRECISION, RECALL AND F1
# FOR XGBOOST
# =========================================

precision = precision_score(
    y_test,
    xgb_prediction,
    average=None,
    zero_division=0
)

recall = recall_score(
    y_test,
    xgb_prediction,
    average=None,
    zero_division=0
)

f1 = f1_score(
    y_test,
    xgb_prediction,
    average=None,
    zero_division=0
)


# =========================================
# ROC CURVE - XGBOOST
# =========================================

n_classes = len(classes)


# Convert actual classes to binary format
y_test_binary = label_binarize(
    y_test,
    classes=range(n_classes)
)


# Create ROC curve
plt.figure(figsize=(8, 6))


for i in range(n_classes):

    fpr, tpr, thresholds = roc_curve(
        y_test_binary[:, i],
        xgb_probability[:, i]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        marker=None,
        label=f"{classes[i]} (AUC = {roc_auc:.2f})"
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
    "XGBoost Multiclass ROC Curve"
)

plt.legend(
    loc="lower right"
)

plt.grid(True)

plt.tight_layout()


plt.savefig(
    "xgboost_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

print("ROC curve saved successfully!")
print("File: xgboost_roc_curve.png")


plt.close()


# =========================================
# CREATE 4 GRAPHS
# =========================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


# =========================================
# GRAPH 1
# ALGORITHM ACCURACY - LINE CHART
# =========================================

algorithms = [
    "Random Forest",
    "Decision Tree",
    "XGBoost",
    "SVM"
]

accuracies = [
    rf_accuracy,
    dt_accuracy,
    xgb_accuracy,
    svm_accuracy
]


axes[0, 0].plot(
    algorithms,
    accuracies,
    marker="o",
    linewidth=2
)


axes[0, 0].set_title(
    "Algorithm Accuracy Comparison"
)

axes[0, 0].set_xlabel(
    "Algorithm"
)

axes[0, 0].set_ylabel(
    "Accuracy"
)

axes[0, 0].set_ylim(
    0,
    1.1
)


# Show accuracy values
for i, value in enumerate(accuracies):

    axes[0, 0].text(
        i,
        value + 0.02,
        f"{value * 100:.2f}%",
        ha="center"
    )


axes[0, 0].grid(True)


# =========================================
# GRAPH 2
# FINANCIAL HEALTH STATUS - PIE CHART
# =========================================

status_names = [
    "Average",
    "Good",
    "Poor"
]


status_counts = df[
    "financial_health_status"
].value_counts()


status_values = [
    status_counts.get("Average", 0),
    status_counts.get("Good", 0),
    status_counts.get("Poor", 0)
]


axes[0, 1].pie(
    status_values,
    labels=status_names,
    autopct="%1.1f%%",
    startangle=90
)


axes[0, 1].set_title(
    "Financial Health Status Distribution"
)

axes[0, 1].axis("equal")


# =========================================
# GRAPH 3
# CONFUSION MATRIX
# =========================================

axes[1, 0].imshow(
    cm
)


axes[1, 0].set_title(
    "Random Forest Confusion Matrix"
)


axes[1, 0].set_xlabel(
    "Predicted Class"
)

axes[1, 0].set_ylabel(
    "Actual Class"
)


axes[1, 0].set_xticks(
    [0, 1, 2]
)

axes[1, 0].set_xticklabels(
    classes
)


axes[1, 0].set_yticks(
    [0, 1, 2]
)

axes[1, 0].set_yticklabels(
    classes
)


# Show confusion matrix values
for i in range(3):

    for j in range(3):

        axes[1, 0].text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


# =========================================
# GRAPH 4
# PRECISION, RECALL AND F1
# =========================================

x = range(
    len(classes)
)

width = 0.25


# Precision
axes[1, 1].bar(
    [i - width for i in x],
    precision,
    width,
    label="Precision"
)


# Recall
axes[1, 1].bar(
    x,
    recall,
    width,
    label="Recall"
)


# F1 Score
axes[1, 1].bar(
    [i + width for i in x],
    f1,
    width,
    label="F1 Score"
)


axes[1, 1].set_title(
    "XGBoost Classification Performance"
)

axes[1, 1].set_xlabel(
    "Financial Health Status"
)

axes[1, 1].set_ylabel(
    "Score"
)


axes[1, 1].set_xticks(
    list(x)
)

axes[1, 1].set_xticklabels(
    classes
)


axes[1, 1].set_ylim(
    0,
    1.1
)


axes[1, 1].legend()


# =========================================
# SAVE ALL 4 GRAPHS
# =========================================

plt.tight_layout()


plt.savefig(
    "all_financial_graphs.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "All 4 graphs saved successfully!"
)

print(
    "File: all_financial_graphs.png"
)
