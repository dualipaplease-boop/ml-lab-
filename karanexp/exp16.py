
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Generate dataset
np.random.seed(42)
df = pd.DataFrame({
    'Annual_Income_k': np.random.randint(15, 135, 200),
    'Spending_Score': np.random.randint(1, 100, 200)
})
print(df.head())

# Preprocess data
X = df.values
X_scaled = StandardScaler().fit_transform(X)

# Apply clustering
k = 5
km = KMeans(n_clusters=k, random_state=42)
km_labels = km.fit_predict(X_scaled)
agg_labels = AgglomerativeClustering(n_clusters=k).fit_predict(X_scaled)

# Evaluate clustering
print("K-Means Inertia:", round(km.inertia_, 2))
print("K-Means Silhouette:", round(silhouette_score(X_scaled, km_labels), 4))
print("Agglomerative Silhouette:", round(silhouette_score(X_scaled, agg_labels), 4))

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, labels, title in zip(
    axes, [km_labels, agg_labels], ['K-Means', 'Agglomerative']
):
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40)
    ax.set_title(title)
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score')

centers = StandardScaler().fit(X).inverse_transform(km.cluster_centers_)
axes[0].scatter(centers[:, 0], centers[:, 1],
                c='black', marker='X', s=150, label='Centroids')
axes[0].legend()
plt.tight_layout()
plt.show()
