# ============================================
# DECISION TREE CLASSIFICATION
# LOAN APPROVAL PREDICTION
# ============================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("07_loan_approval.csv")

print("========== DATASET ==========")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================
# 2. CHECK MISSING VALUES
# ============================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ============================================
# 3. TARGET VARIABLE DISTRIBUTION
# ============================================

print("\n========== LOAN APPROVAL DISTRIBUTION ==========")
print(df["LoanApproved"].value_counts())


# ============================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================

# ApplicantID is removed because it is only an ID
X = df.drop(columns=["LoanApproved", "ApplicantID"])

# Target variable
y = df["LoanApproved"]


# ============================================
# 5. DEFINE CATEGORICAL AND NUMERICAL COLUMNS
# ============================================

categorical_columns = [
    "Education",
    "MaritalStatus",
    "PropertyArea",
    "SelfEmployed"
]

numerical_columns = [
    "Age",
    "AnnualIncome",
    "LoanAmount",
    "CreditScore",
    "EmploymentYears"
]


# ============================================
# 6. ONE-HOT ENCODING
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ============================================
# 7. CREATE DECISION TREE CLASSIFIER
# ============================================

classifier = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)


# ============================================
# 8. CREATE PIPELINE
# ============================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


# ============================================
# 9. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 10. TRAIN THE MODEL
# ============================================

model.fit(X_train, y_train)

print("\nModel training completed successfully.")


# ============================================
# 11. PREDICTION
# ============================================

y_pred = model.predict(X_test)


# ============================================
# 12. ACCURACY
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")


# ============================================
# 13. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["N", "Y"]
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)


# ============================================
# 14. CLASSIFICATION REPORT
# ============================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["N", "Y"]
    )
)


# ============================================
# 15. DISPLAY CONFUSION MATRIX
# ============================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["N", "Y"]
)

disp.plot()

plt.title("Confusion Matrix - Decision Tree")
plt.tight_layout()
plt.show()


# ============================================
# 16. GET FEATURE NAMES
# ============================================

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


# ============================================
# 17. DECISION TREE VISUALIZATION
# ============================================

plt.figure(figsize=(25, 12))

plot_tree(
    model.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["N", "Y"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - Loan Approval Prediction")
plt.tight_layout()
plt.show()


# ============================================
# 18. FEATURE IMPORTANCE
# ============================================

feature_importance = pd.Series(
    model.named_steps["classifier"].feature_importances_,
    index=feature_names
)

feature_importance = feature_importance.sort_values(
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)


# ============================================
# 19. FEATURE IMPORTANCE GRAPH
# ============================================

plt.figure(figsize=(10, 6))

feature_importance.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()
plt.show()


# ============================================
# 20. PREDICT NEW APPLICANT
# ============================================

new_applicant = pd.DataFrame({
    "Age": [30],
    "AnnualIncome": [60000],
    "LoanAmount": [80000],
    "CreditScore": [750],
    "EmploymentYears": [5],
    "Education": ["Graduate"],
    "MaritalStatus": ["Married"],
    "PropertyArea": ["Urban"],
    "SelfEmployed": ["No"]
})

prediction = model.predict(new_applicant)

print("\n========== NEW APPLICANT PREDICTION ==========")

if prediction[0] == "Y":
    print("Loan Approved")
else:
    print("Loan Rejected")


# ============================================
# 21. PREDICTION PROBABILITY
# ============================================

probability = model.predict_proba(new_applicant)

print("\nPrediction Probability:")

classes = model.named_steps["classifier"].classes_

for class_name, prob in zip(classes, probability[0]):
    print(class_name, ":", prob)


# ============================================
# END
# ============================================

print("\n========== PRACTICAL COMPLETED ==========")