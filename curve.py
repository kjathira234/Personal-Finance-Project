import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# =========================================
# LOAD GRAPH RESULTS
# =========================================

results = joblib.load("graph_results.pkl")


# =========================================
# GET DATA
# =========================================

y_test = results["y_test"]

rf_probability = results["rf_probability"]


# =========================================
# FINANCIAL HEALTH CLASSES
# =========================================

classes = [
    "Average",
    "Good",
    "Poor"
]

n_classes = len(classes)


# =========================================
# CONVERT CLASSES TO BINARY
# =========================================

y_test_binary = label_binarize(
    y_test,
    classes=[0, 1, 2]
)


# =========================================
# RANDOM FOREST ROC CURVE
# =========================================

plt.figure(figsize=(8, 6))


for i in range(n_classes):

    fpr, tpr, thresholds = roc_curve(
        y_test_binary[:, i],
        rf_probability[:, i]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{classes[i]} (AUC = {roc_auc:.2f})"
    )


# =========================================
# RANDOM CLASSIFIER LINE
# =========================================

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


# =========================================
# LABELS
# =========================================

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Random Forest Multiclass ROC Curve"
)


# =========================================
# LEGEND
# =========================================

plt.legend(
    loc="lower right"
)

plt.grid(True)


# =========================================
# SAVE ROC CURVE
# =========================================

plt.tight_layout()

plt.savefig(
    "random_forest_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

print("Random Forest ROC curve saved successfully!")

print(
    "File: random_forest_roc_curve.png"
)


# =========================================
# SHOW GRAPH
# =========================================

plt.show()