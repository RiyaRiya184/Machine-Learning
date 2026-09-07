# Machine-Learning
# 1. FAOSTAT Wheat Yield Prediction

## Objective

To analyze FAOSTAT wheat crop data and predict wheat yield using Linear Regression, Polynomial Regression, and Multivariate Regression.

## Dataset

The dataset is obtained from FAOSTAT and contains wheat data with the following elements:

- Area harvested
- Production
- Yield

The important columns used are:

- Year
- Yield
- Production
- Area

## Data Preprocessing

The dataset was filtered to include only Wheat.

The FAOSTAT `Element` column was used to separate:

- Yield
- Production
- Area harvested

The three datasets were then merged using Year.

The data was divided into:

- 70% Training Data
- 30% Testing Data

## 1. Linear Regression

Linear Regression was used to predict wheat yield using Year as the independent variable.

### Results

| Metric | Value |
|---|---:|
|MSE |20739.265753491687|
|RMSE | 144.0113389754143|
|MAE  | 119.55594176884934|
|R2   | 0.9770967958664648|


### Graph

![Linear Regression Graph](https://github.com/RiyaRiya184/Machine-Learning/blob/b06461ad943e6ad744766c3eeb2b34347a8b6809/Images/Linear%20regression%20graph.png)

---

## 2. Polynomial Regression

Polynomial Regression with degree 2 was used to model the relationship between Year and wheat yield.

### Results

| Metric | Value |
|---|---:|
| MSE |  18311.838329379363|
| RMSE |135.32124123499372 |
| MAE | 119.5050655174442 |
| R² |  0.9770967958664648|

### Graph

![Polynomial Regression Graph](https://github.com/RiyaRiya184/Machine-Learning/blob/ee7a1ee2772ec8276a3b3c5e2a45267924bac4c8/Images/polynomial%20regression%20graph.png)

---

## 3. Multivariate Regression

Multivariate Regression was used to predict wheat yield using:

- Year
- Production
- Area harvested

### Results

| Metric | Value |
|---|---:|
| MSE | 9648.939904768764  |
| RMSE | 98.22901763108885 |
| MAE | 81.0383099083562 |
| R² |  0.9893442881277547 |

### Graph

![Multivariate Regression Graph](https://github.com/RiyaRiya184/Machine-Learning/blob/de4610d74da28d274ffb273db50d3199091a14d2/Images/Multivariate%20regression%20graph.png)

---

## Model Comparison

| Model | MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 20739.265753 | 144.011339 | 119.555942  |0.977097  | 
| Polynomial Regression |18311.838329 | 135.321241 | 119.505066 | 0.979778 |
| Multivariate Regression | 9648.939905 |  98.229018 |  81.038310 | 0.989344  |

## Future Prediction

The model was also used to predict wheat yield for **2030**.

**Predicted Yield:** 3850.77943288158

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Conclusion

The three regression models were evaluated using MSE, RMSE, MAE, and R². The model with the best performance can be selected based on the evaluation metrics, particularly R² and error values.
---

# 2. Decision Tree – Loan Approval Prediction

## Aim
To implement a Decision Tree Classification algorithm using Python to predict whether a loan application will be approved or rejected.

## Dataset
The dataset `07_loan_approval.csv` contains 600 loan applications with the following features:

- Age
- Annual Income
- Loan Amount
- Credit Score
- Employment Years
- Education
- Marital Status
- Property Area
- Self Employed

Target variable: `LoanApproved`

- `Y` = Loan Approved
- `N` = Loan Rejected

`ApplicantID` was excluded from the model because it is only an identifier.

## Methodology

1. Load the loan approval dataset.
2. Separate the input features and target variable.
3. Encode categorical features using One-Hot Encoding.
4. Split the dataset into 70% training and 30% testing data.
5. Train a Decision Tree Classifier using Entropy.
6. Set the maximum tree depth to 5 to reduce overfitting.
7. Predict loan approval on the test dataset.
8. Evaluate the model using accuracy, confusion matrix, precision, recall and F1-score.
9. Visualize the decision tree and feature importance.

## Algorithm

The Decision Tree algorithm recursively splits the dataset using the feature that provides the highest Information Gain.

### Entropy

Entropy measures the impurity of a dataset.

`Entropy(S) = -Σ pᵢ log₂(pᵢ)`

### Information Gain

`Information Gain = Entropy(parent) - Weighted Entropy(children)`

The first major split of the trained tree was based on **Credit Score**.

## Model Configuration

- Algorithm: Decision Tree Classifier
- Criterion: Entropy
- Maximum Depth: 5
- Training Data: 70%
- Testing Data: 30%
- Random State: 42

## Results

- Total Records: 600
- Training Records: 420
- Testing Records: 180
- Accuracy: **66.11%**

### Confusion Matrix

| Actual / Predicted | N | Y |
|---|---:|---:|
| N | 62 | 38 |
| Y | 23 | 57 |

The model correctly classified 119 out of 180 test samples.

## Visualizations

### Confusion Matrix

![Confusion Matrix](https://github.com/RiyaRiya184/Machine-Learning/blob/c164798a96bd99bd6c5f13049fa5ae3066847e50/Images/Confusion_matrix.png)

### Decision Tree

![Decision Tree](https://github.com/RiyaRiya184/Machine-Learning/blob/c164798a96bd99bd6c5f13049fa5ae3066847e50/Images/Decision_tree.png)

### Feature Importance

![Feature Importance](https://github.com/RiyaRiya184/Machine-Learning/blob/c164798a96bd99bd6c5f13049fa5ae3066847e50/Images/Imp_Features.png)

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Conclusion

The Decision Tree Classification algorithm was successfully implemented to predict loan approval. The model achieved an accuracy of **66.11%** on the test dataset. The practical demonstrates the use of Entropy, Information Gain, Decision Tree Classification, data preprocessing and model evaluation for a real-world loan approval prediction problem.
---
# 3. Naive Bayes Classification - Mushroom Edibility Dataset

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

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("11_mushroom_edibility.csv")

print("=" * 60)
print("MUSHROOM DATASET")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. Dataset Information
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())


# --------------------------------------------------
# 3. Separate Features and Target
# --------------------------------------------------

# Remove SampleID because it is only an identifier
X = df.drop(columns=["Class", "SampleID"])

# Target variable
y = df["Class"]

print("\nFeatures Used:")
print(X.columns.tolist())

print("\nTarget Variable:")
print("Class")


# --------------------------------------------------
# 4. Encode Target Variable
# --------------------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\n" + "=" * 60)
print("TARGET ENCODING")
print("=" * 60)

for class_name, encoded_value in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(class_name, "=", encoded_value)


# --------------------------------------------------
# 5. Train-Test Split
# --------------------------------------------------

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


# --------------------------------------------------
# 6. Encode Categorical Features
# --------------------------------------------------

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

X_train_encoded = X_train_encoded.astype(int)
X_test_encoded = X_test_encoded.astype(int)


# --------------------------------------------------
# 7. Train Naive Bayes Model
# --------------------------------------------------

model = CategoricalNB()

model.fit(X_train_encoded, y_train)

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

print("Algorithm: Categorical Naive Bayes")
print("Model trained successfully!")


# --------------------------------------------------
# 8. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test_encoded)

y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)


# --------------------------------------------------
# 9. Calculate Performance Metrics
# --------------------------------------------------

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


# --------------------------------------------------
# 10. Classification Report
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test_labels,
        y_pred_labels
    )
)


# --------------------------------------------------
# 11. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test_labels,
    y_pred_labels
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot()

plt.title("Naive Bayes - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "naive_bayes_confusion_matrix.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 12. Actual vs Predicted Values
# --------------------------------------------------

comparison = pd.DataFrame({
    "Actual": y_test_labels,
    "Predicted": y_pred_labels
})

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED VALUES")
print("=" * 60)

print(comparison.head(20))


# --------------------------------------------------
# 13. Mushroom Class Distribution Graph
# --------------------------------------------------

plt.figure(figsize=(7, 5))

df["Class"].value_counts().plot(
    kind="bar"
)

plt.title("Mushroom Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Mushrooms")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "mushroom_class_distribution.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 14. Performance Metrics Graph
# --------------------------------------------------

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

for i, value in enumerate(metrics.values()):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "naive_bayes_performance.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 15. Sample Predictions
# --------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

sample_predictions = model.predict(
    X_test_encoded[:5]
)

sample_predictions = label_encoder.inverse_transform(
    sample_predictions
)

for i, prediction in enumerate(sample_predictions):
    print(
        f"Sample {i + 1}: Predicted class = {prediction}"
    )


# --------------------------------------------------
# 16. Final Result
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(
    f"Naive Bayes achieved an accuracy of "
    f"{accuracy * 100:.2f}% on the test dataset."
)

print("\nPractical completed successfully!")
---
