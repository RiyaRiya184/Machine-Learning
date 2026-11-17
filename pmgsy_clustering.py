"""
PMGSY State Typology — reproducible analysis
Input: pmgsy_state_year_data.csv (same folder)
Install: pip install pandas numpy scikit-learn matplotlib
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("pmgsy_state_year_data.csv")
year_cols = ["2021-22", "2022-23", "2023-24"]
Xtrend = np.arange(len(year_cols)).reshape(-1, 1)
df["Trend_slope_km_per_year"] = [
    LinearRegression().fit(Xtrend, row[year_cols].astype(float).values).coef_[0]
    for _, row in df.iterrows()
]
features = df[year_cols + ["Trend_slope_km_per_year"]]
X = StandardScaler().fit_transform(features)

scores = {}
for k in range(2, 6):
    labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
    scores[k] = silhouette_score(X, labels)
best_k = max(scores, key=scores.get)

km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["KMeans_cluster"] = km.fit_predict(X) + 1
df["Hierarchical_cluster"] = (
    AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X) + 1
)
print("Silhouette scores by k:", scores)
print("Selected k:", best_k)
print("K-means silhouette:", silhouette_score(X, df["KMeans_cluster"]))
print("Hierarchical silhouette:", silhouette_score(X, df["Hierarchical_cluster"]))
print(df.sort_values(["KMeans_cluster", "Total_3yr_km"]).to_string(index=False))
df.to_csv("pmgsy_clustered_results.csv", index=False)

plt.figure(figsize=(9, 5))
for c, g in df.groupby("KMeans_cluster"):
    plt.scatter(g["Mean_annual_km"], g["Trend_slope_km_per_year"], label=f"Cluster {c}")
plt.xlabel("Mean annual completed road length (km)")
plt.ylabel("Trend slope (km/year)")
plt.title("PMGSY state typology using K-means")
plt.legend()
plt.tight_layout()
plt.savefig("pmgsy_cluster_plot.png", dpi=180)
