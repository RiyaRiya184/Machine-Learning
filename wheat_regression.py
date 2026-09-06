import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. READ FAOSTAT DATA
# ============================================================

df = pd.read_csv("wheat.csv")

print("File loaded successfully!")
print("Original rows:", len(df))


# ============================================================
# 2. FILTER ONLY WHEAT
# ============================================================

wheat = df[df["Item"] == "Wheat"].copy()

print("Wheat rows:", len(wheat))


# ============================================================
# 3. SEPARATE YIELD, PRODUCTION AND AREA
# ============================================================

yield_data = wheat[wheat["Element"] == "Yield"][["Year", "Value"]]

production_data = wheat[wheat["Element"] == "Production"][["Year", "Value"]]

area_data = wheat[wheat["Element"] == "Area harvested"][["Year", "Value"]]


# Rename Value column

yield_data = yield_data.rename(columns={"Value": "Yield"})

production_data = production_data.rename(columns={"Value": "Production"})

area_data = area_data.rename(columns={"Value": "Area"})


# ============================================================
# 4. COMBINE THE DATA USING YEAR
# ============================================================

data = yield_data.merge(production_data, on="Year")

data = data.merge(area_data, on="Year")


# ============================================================
# 5. CLEAN DATA
# ============================================================

data = data.dropna()

data = data.sort_values("Year")

print("\nFinal dataset:")
print(data.head())

print("\nNumber of rows:", len(data))

print("\nMissing values:")
print(data.isnull().sum())


# ============================================================
# 6. LINEAR REGRESSION
#    Predict Yield using Year
# ============================================================

X = data[["Year"]]

y = data["Yield"]


# 70% Training, 30% Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# Create and train model

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)


# Predict

y_pred_linear = linear_model.predict(X_test)


# Metrics

mse_linear = mean_squared_error(y_test, y_pred_linear)

rmse_linear = np.sqrt(mse_linear)

mae_linear = mean_absolute_error(y_test, y_pred_linear)

r2_linear = r2_score(y_test, y_pred_linear)


print("\n====================================")
print("LINEAR REGRESSION")
print("====================================")

print("MSE  :", mse_linear)
print("RMSE :", rmse_linear)
print("MAE  :", mae_linear)
print("R2   :", r2_linear)


# ============================================================
# 7. LINEAR REGRESSION GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(X_test["Year"], y_test, label="Actual")

plt.plot(
    X_test["Year"],
    y_pred_linear,
    label="Predicted"
)

plt.xlabel("Year")

plt.ylabel("Wheat Yield")

plt.title("Linear Regression - Wheat Yield")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 8. POLYNOMIAL REGRESSION
# ============================================================

X = data[["Year"]]

y = data["Yield"]


# Create polynomial features

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)


# 70% Training, 30% Testing

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly,
    y,
    test_size=0.30,
    random_state=42
)


# Create model

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train_poly)


# Predict

y_pred_poly = poly_model.predict(X_test_poly)


# Metrics

mse_poly = mean_squared_error(y_test_poly, y_pred_poly)

rmse_poly = np.sqrt(mse_poly)

mae_poly = mean_absolute_error(y_test_poly, y_pred_poly)

r2_poly = r2_score(y_test_poly, y_pred_poly)


print("\n====================================")
print("POLYNOMIAL REGRESSION")
print("====================================")

print("MSE  :", mse_poly)
print("RMSE :", rmse_poly)
print("MAE  :", mae_poly)
print("R2   :", r2_poly)


# ============================================================
# 9. POLYNOMIAL REGRESSION GRAPH
# ============================================================

# Create smooth year values for graph

years = np.linspace(
    data["Year"].min(),
    data["Year"].max(),
    200
).reshape(-1, 1)


years_poly = poly.transform(years)

predicted_yield = poly_model.predict(years_poly)


plt.figure(figsize=(8, 5))

plt.scatter(
    data["Year"],
    data["Yield"],
    label="Actual"
)

plt.plot(
    years,
    predicted_yield,
    label="Polynomial Prediction"
)

plt.xlabel("Year")

plt.ylabel("Wheat Yield")

plt.title("Polynomial Regression - Wheat Yield")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 10. MULTIVARIATE REGRESSION
#
# Predict Yield using:
# Year
# Production
# Area
# ============================================================

X = data[["Year", "Production", "Area"]]

y = data["Yield"]


# 70% Training, 30% Testing

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# Create model

multi_model = LinearRegression()

multi_model.fit(X_train_multi, y_train_multi)


# Predict

y_pred_multi = multi_model.predict(X_test_multi)


# Metrics

mse_multi = mean_squared_error(y_test_multi, y_pred_multi)

rmse_multi = np.sqrt(mse_multi)

mae_multi = mean_absolute_error(y_test_multi, y_pred_multi)

r2_multi = r2_score(y_test_multi, y_pred_multi)


print("\n====================================")
print("MULTIVARIATE REGRESSION")
print("====================================")

print("MSE  :", mse_multi)
print("RMSE :", rmse_multi)
print("MAE  :", mae_multi)
print("R2   :", r2_multi)


# ============================================================
# 11. MULTIVARIATE REGRESSION GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test_multi,
    y_pred_multi,
    label="Predicted vs Actual"
)

# Perfect prediction line

minimum = min(y_test_multi.min(), y_pred_multi.min())

maximum = max(y_test_multi.max(), y_pred_multi.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    label="Perfect Prediction"
)

plt.xlabel("Actual Wheat Yield")

plt.ylabel("Predicted Wheat Yield")

plt.title("Multivariate Regression - Wheat Yield")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Multivariate Regression"
    ],

    "MSE": [
        mse_linear,
        mse_poly,
        mse_multi
    ],

    "RMSE": [
        rmse_linear,
        rmse_poly,
        rmse_multi
    ],

    "MAE": [
        mae_linear,
        mae_poly,
        mae_multi
    ],

    "R2": [
        r2_linear,
        r2_poly,
        r2_multi
    ]
})


print("\n====================================")
print("MODEL COMPARISON")
print("====================================")

print(results)


# ============================================================
# 13. PREDICT FUTURE WHEAT YIELD
# ============================================================

future_year = 2030

future_prediction = linear_model.predict(
    [[future_year]]
)


print("\n====================================")
print("FUTURE PREDICTION")
print("====================================")

print(
    "Predicted wheat yield for",
    future_year,
    ":",
    future_prediction[0]
)


# ============================================================
# 14. POLYNOMIAL FUTURE PREDICTION
# ============================================================

future_year_poly = poly.transform(
    [[future_year]]
)

future_prediction_poly = poly_model.predict(
    future_year_poly
)


print(
    "Polynomial predicted wheat yield for",
    future_year,
    ":",
    future_prediction_poly[0]
)