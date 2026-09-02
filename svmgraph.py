import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay(
    confusion_matrix=svm_cm,
    display_labels=target_encoder.classes_
).plot()

plt.title("SVM Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "svm_confusion_matrix.png",
    dpi=300
)

plt.show()