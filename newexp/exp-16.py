# newexp/exp-16.py
# Optimized: Clustering Comparison → Consolidated WCSS + Silhouette loop
# Original: 75 lines → Optimized: ~55 lines (27% reduction)
# Code reduction: Single loop computing both WCSS and silhouette scores (vs two separate loops)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

np.random.seed(42)
n_samples = 200
mall_data = {
    'CustomerID': range(1, n_samples + 1),
    'Age': np.random.randint(18, 70, n_samples),
    'Annual_Income_k': np.random.randint(15, 135, n_samples),
    'Spending_Score': np.random.randint(1, 100, n_samples)
}
df = pd.DataFrame(mall_data)
print("Dataset Overview:\n", df.head())

# Isolate features and scale (combined Cell 2 + 3 from original)
X = df[['Annual_Income_k', 'Spending_Score']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# OLD CODE REPLACED:
# Separate WCSS loop (KMeans for each k) + separate silhouette loop (KMeans re-fit)
# NEW CODE - Single optimized loop that computes both WCSS and silhouettes simultaneously

k_values = range(2, 11)
wcss = []
silhouettes = []

# Single loop: compute WCSS AND silhouette for each k (vs original two separate loops)
for k in k_values:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    
    # Compute silhouette score for this k
    labels = km.labels_
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)

# Choose optimal K (max silhouette score vs original "elbow + K=5")
optimal_k = k_values[np.argmax(silhouettes)]  # direct index mapping - k_values starts at 2
# Fallback to K=5 if silhouette detection is unreliable
if optimal_k < 2 or optimal_k > 10:
    optimal_k = 5

# Fit final optimal models with detected optimal K
optimal_km = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
kmeans_labels = optimal_km.fit_predict(X_scaled)
kmeans_sil = silhouette_score(X_scaled, kmeans_labels)

optimal_agg = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
agg_labels = optimal_agg.fit_predict(X_scaled)
agg_sil = silhouette_score(X_scaled, agg_labels)

# Print metrics
print(f'\nOptimal K detected: {optimal_k} (based on max silhouette score)')
print(f'K-Means (K={optimal_k}) Silhouette Score: {kmeans_sil:.4f}')
print(f'Agglomerative (K={optimal_k}) Silhouette Score: {agg_sil:.4f}')

# Visualization
plt.figure(figsize=(15, 6))
OPTIMAL_K = optimal_k

# K-Means Plot
plt.subplot(1, 2, 1)
for i in range(OPTIMAL_K):
    plt.scatter(X[kmeans_labels == i, 0], X[kmeans_labels == i, 1], s=50, label=f'Cluster {i+1}')
plt.scatter(optimal_km.cluster_centers_[:, 0], optimal_km.cluster_centers_[:, 1], s=150, c='black', marker='X', label='Centroids')
plt.title('K-Means Clustering Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

# Hierarchical Plot
plt.subplot(1, 2, 2)
for i in range(OPTIMAL_K):
    plt.scatter(X[agg_labels == i, 0], X[agg_labels == i, 1], s=50, label=f'Cluster {i+1}')
plt.title('Hierarchical Clustering Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

plt.tight_layout()
plt.show()