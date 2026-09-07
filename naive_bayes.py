# ============================================================
# NAIVE BAYES CLASSIFICATION - MUSHROOM EDIBILITY DATASET
# ============================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

file_path = "./datasets/11_mushroom_edibility.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("MUSHROOM DATASET")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 2. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

# ------------------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

# SampleID is only an identifier, so it is removed
X = df.drop(columns=["Class", "SampleID"])

# Target variable
y = df["Class"]

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget:")
print("Class")

# ------------------------------------------------------------
# 4. ENCODE TARGET VARIABLE
# ------------------------------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget encoding:")
for class_name, encoded_value in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(class_name, "=", encoded_value)

# ------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 6. ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------

# CategoricalNB requires categorical values to be represented
# as integer values starting from 0.

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Convert to integer
X_train_encoded = X_train_encoded.astype(int)
X_test_encoded = X_test_encoded.astype(int)

# ------------------------------------------------------------
# 7. TRAIN NAIVE BAYES MODEL
# ------------------------------------------------------------

model = CategoricalNB()

model.fit(X_train_encoded, y_train)

print("\n" + "=" * 60)
print("MODEL TRAINED")
print("=" * 60)

print("Algorithm: Categorical Naive Bayes")

# ------------------------------------------------------------
# 8. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test_encoded)

# Convert predictions back to original class names
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# ------------------------------------------------------------
# 9. MODEL EVALUATION
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ------------------------------------------------------------
# 10. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test_labels,
        y_pred_labels
    )
)

# ------------------------------------------------------------
# 11. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test_labels, y_pred_labels)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot()

plt.title("Naive Bayes - Confusion Matrix")
plt.tight_layout()

# Save graph
plt.savefig("naive_bayes_confusion_matrix.png", dpi=300)

plt.show()

# ------------------------------------------------------------
# 12. ACTUAL VS PREDICTED VALUES
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "Actual": y_test_labels,
    "Predicted": y_pred_labels
})

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED")
print("=" * 60)

print(comparison.head(20))

# ------------------------------------------------------------
# 13. CLASS DISTRIBUTION GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

df["Class"].value_counts().plot(
    kind="bar"
)

plt.title("Mushroom Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Mushrooms")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("mushroom_class_distribution.png", dpi=300)

plt.show()

# ------------------------------------------------------------
# 14. ACCURACY METRICS GRAPH
# ------------------------------------------------------------

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

plt.figure(figsize=(8, 5))

plt.bar(
    metrics.keys(),
    metrics.values()
)

plt.ylim(0, 1)

plt.title("Naive Bayes Performance Metrics")
plt.xlabel("Metrics")
plt.ylabel("Score")

# Add values on top of bars
for i, value in enumerate(metrics.values()):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig("naive_bayes_performance.png", dpi=300)

plt.show()

# ------------------------------------------------------------
# 15. PREDICTION EXAMPLE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE PREDICTION")
print("=" * 60)

# Take first 5 test samples
sample_predictions = model.predict(X_test_encoded[:5])

sample_predictions = label_encoder.inverse_transform(
    sample_predictions
)

for i, prediction in enumerate(sample_predictions):
    print(
        f"Sample {i + 1}: Predicted class = {prediction}"
    )

# ------------------------------------------------------------
# 16. FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(
    f"Naive Bayes achieved an accuracy of "
    f"{accuracy * 100:.2f}% on the test dataset."
)

print("\nPractical completed successfully.")