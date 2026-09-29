import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# Load dataset
df = pd.read_csv("07_loan_approval.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# Separate features and target
X = df.drop(columns=["ApplicantID", "LoanApproved"])
y = df["LoanApproved"].map({"Y": 1, "N": 0})


# Numerical and categorical columns
numerical_features = [
    "Age",
    "AnnualIncome",
    "LoanAmount",
    "CreditScore",
    "EmploymentYears"
]

categorical_features = [
    "Education",
    "MaritalStatus",
    "PropertyArea",
    "SelfEmployed"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create SVM model
svm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])


# Train model
svm_model.fit(X_train, y_train)


# Prediction
y_pred = svm_model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nSVM Accuracy:", accuracy)


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Approved", "Approved"]
)

disp.plot()
plt.title("SVM Confusion Matrix")
plt.show()