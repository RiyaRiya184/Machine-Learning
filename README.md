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
---
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
---
---
# 3. Naive Bayes Classification - Mushroom Edibility Dataset
# Naive Bayes Classification - Mushroom Edibility

## Aim

To implement the Naive Bayes classification algorithm on a Mushroom Edibility dataset and predict whether a mushroom is **edible or poisonous**.

## Dataset

The dataset used for this practical is:

`11_mushroom_edibility.csv`

The dataset contains mushroom characteristics and a target variable representing mushroom edibility.

### Features Used

- CapShape
- CapColor
- Odor
- GillSpacing
- GillColor
- StalkShape
- RingNumber
- Habitat
- Population

### Target Variable

`Class`

The target classes are:

- Edible
- Poisonous

`SampleID` is removed because it is only an identifier and does not contribute to the prediction.

## Theory

Naive Bayes is a supervised machine learning classification algorithm based on Bayes' Theorem.

It calculates the probability of a class based on the given features.

The algorithm assumes that the features are conditionally independent given the class.

For classification:

P(Class | Features) ∝ P(Class) × P(Feature1 | Class) × P(Feature2 | Class) × ... × P(FeatureN | Class)

## Algorithm Used

### Categorical Naive Bayes

Since the features in the Mushroom dataset are categorical, **Categorical Naive Bayes (`CategoricalNB`)** is used.

Categorical features are converted into numerical values using `OrdinalEncoder`, while the target variable is encoded using `LabelEncoder`.

## Methodology

### 1. Load the Dataset

The dataset is loaded using Pandas.

### 2. Data Preprocessing

- Check the dataset shape.
- Check column names and data types.
- Check for missing values.
- Remove `SampleID`.
- Separate input features and target variable.
- Encode categorical features.

### 3. Train-Test Split

The dataset is divided into:

- 70% training data
- 30% testing data

A stratified split is used so that the class distribution is maintained in both training and testing sets.

### 4. Model Training

A Categorical Naive Bayes classifier is trained using the training dataset.

### 5. Prediction

The trained model predicts whether the mushrooms in the test dataset are edible or poisonous.

### 6. Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

## Results

The program calculates the following performance metrics:

| Metric | Result |
|---|---:|
| Accuracy | 0.7556 |
| Precision | 0.7548 |
| Recall | 0.7556 |
| F1 Score | 0.7551 |

## Visualizations

### Mushroom Class Distribution

The class distribution graph shows the number of edible and poisonous mushrooms present in the dataset.

![Mushroom Class Distribution](https://github.com/RiyaRiya184/Machine-Learning/blob/91b7690e13955cfbaba250eb74cdc6769e032f98/Images/Mushroom%20class%20distribution.png)

### Confusion Matrix

The confusion matrix shows the correctly and incorrectly classified mushroom samples.

![Naive Bayes Confusion Matrix](https://github.com/RiyaRiya184/Machine-Learning/blob/91b7690e13955cfbaba250eb74cdc6769e032f98/Images/Naive%20bayes%20confusion%20matrix.png)

### Performance Metrics

The performance graph displays the Accuracy, Precision, Recall and F1 Score of the Naive Bayes model.

![Naive Bayes Performance](https://github.com/RiyaRiya184/Machine-Learning/blob/91b7690e13955cfbaba250eb74cdc6769e032f98/Images/Naive%20bayes%20performance%20metrics.png)

## Confusion Matrix

The confusion matrix consists of:

- **True Positive (TP):** Correctly predicted positive class.
- **True Negative (TN):** Correctly predicted negative class.
- **False Positive (FP):** Negative sample incorrectly classified as positive.
- **False Negative (FN):** Positive sample incorrectly classified as negative.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Libraries Used

```python
pandas
numpy
matplotlib
scikit-learn

---
---
---
# 2. K-Nearest Neighbors (KNN) Classification

## Objective

To implement the K-Nearest Neighbors (KNN) algorithm and classify a new data point using different values of K.

## Dataset

The dataset contains six points with two features and their respective classes:

| Point | X1 | X2 | Class |
|------|----|----|-------|
| P1 | 4 | 3 | B |
| P2 | 3 | 3 | A |
| P3 | 5 | 5 | A |
| P4 | 2 | 4 | A |
| P5 | 8 | 8 | B |
| P6 | 7 | 2 | B |

The query point is:

**Q = (4,4)**

## Distance Calculation

Euclidean distance was used to calculate the distance between the query point and each training point.

The points were sorted based on their distance from Q.

| Point | Distance | Class |
|------|----------|-------|
| P1 | 1.000 | B |
| P2 | 1.414 | A |
| P3 | 1.414 | A |
| P4 | 2.000 | A |
| P6 | 3.606 | B |
| P5 | 5.657 | B |

## KNN Classification

The query point was classified using:

- K = 1
- K = 3
- K = 5

### Results

| K | Predicted Class |
|---|-----------------|
| 1 | B |
| 3 | A |
| 5 | A |

## Python Code

```python
import math

data = [
    (4, 3, 'B'),
    (3, 3, 'A'),
    (5, 5, 'A'),
    (2, 4, 'A'),
    (8, 8, 'B'),
    (7, 2, 'B')
]

Q = (4, 4)

distances = []

for x1, x2, cls in data:
    d = math.sqrt((x1 - Q[0])**2 + (x2 - Q[1])**2)
    distances.append((d, cls))

distances.sort()

for k in [1, 3, 5]:
    neighbors = distances[:k]

    A = sum(cls == 'A' for d, cls in neighbors)
    B = sum(cls == 'B' for d, cls in neighbors)

    prediction = 'A' if A > B else 'B'

    print(f"K = {k} → Class = {prediction}")
```

## Output

```text
K = 1 → Class = B
K = 3 → Class = A
K = 5 → Class = A
```

## Conclusion

The KNN algorithm was successfully implemented to classify the query point **Q = (4,4)**. The classification changes with the value of K. For K = 1, the predicted class is **B**, while for K = 3 and K = 5, the predicted class is **A**.
---
---
---
