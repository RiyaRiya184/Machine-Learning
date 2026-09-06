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

