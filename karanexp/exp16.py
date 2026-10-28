import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv("/content/Mall_Customers.csv")
print(df.head())

# Select and scale features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
km = KMeans(n_clusters=5, random_state=42, n_init=10)
y_km = km.fit_predict(X_scaled)

# Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=5)
y_hc = hc.fit_predict(X_scaled)

# Evaluation
print("Inertia:", round(km.inertia_, 2))
print("K-Means Silhouette:", round(silhouette_score(X_scaled, y_km), 4))
print("Agglomerative Silhouette:", round(silhouette_score(X_scaled, y_hc), 4))

# Convert centroids to original scale
centroids = scaler.inverse_transform(km.cluster_centers_)

# Plot clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(X[:, 0], X[:, 1], c=y_km, cmap='viridis', s=50)
ax1.scatter(centroids[:, 0], centroids[:, 1],
            c='red', marker='X', s=150, label='Centroids')
ax1.set_title('K-Means (K=5)')
ax1.set_xlabel('Annual Income (k$)')
ax1.set_ylabel('Spending Score')
ax1.legend()
ax1.grid()

ax2.scatter(X[:, 0], X[:, 1], c=y_hc, cmap='viridis', s=50)
ax2.set_title('Agglomerative (K=5)')
ax2.set_xlabel('Annual Income (k$)')
ax2.set_ylabel('Spending Score')
ax2.grid()

plt.tight_layout()
plt.show()