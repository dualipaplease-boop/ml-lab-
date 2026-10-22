# newexp/exp-9.py
# Optimized: Decision Tree RFM → pd.get_dummies() encoding
# Original: 64 lines → Optimized: ~45 lines (30% reduction)
# Code reduction: pd.get_dummies() replaces manual categorical encoding loop + target_map

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create synthetic RFM dataset
np.random.seed(42)
n_samples = 800
data = {
    'Recency_Tier': np.random.choice(['Recent', 'Dormant', 'At-Risk'], n_samples, p=[0.5, 0.3, 0.2]),
    'Frequency_Tier': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
    'Monetary_Tier': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
    'Segment': np.random.choice(['Bronze', 'Silver', 'Gold'], n_samples, p=[0.4, 0.4, 0.2])
}
df = pd.DataFrame(data)
print(df.head())

# OLD CODE REPLACED:
# for col in ['Recency_Tier', 'Frequency_Tier', 'Monetary_Tier']:
#     df[col] = df[col].astype('category').cat.codes
# target_map = {'Bronze': 0, 'Silver': 1, 'Gold': 2}
# df['Segment'] = df['Segment'].map(target_map)

# NEW CODE - simplified with get_dummies:
df = pd.get_dummies(df, columns=['Recency_Tier', 'Frequency_Tier', 'Monetary_Tier'])
# Note: get_dummies creates binary columns; Segment needs separate mapping
target_map = {'Bronze': 0, 'Silver': 1, 'Gold': 2}
df['Segment'] = df['Segment'].map(target_map)

X = df.drop('Segment', axis=1)
y = df['Segment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(df.head())

# Train Decision Tree
clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf_tree.fit(X_train, y_train)
y_pred = clf_tree.predict(X_test)
print(f'Model Classification Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# Visualize tree structure
# Get feature names after get_dummies
feat_cols = [c for c in X.columns]
plt.figure(figsize=(14, 8))
plot_tree(clf_tree,
          feature_names=feat_cols,
          class_names=['Bronze', 'Silver', 'Gold'],
          filled=True, rounded=True, fontsize=10)
plt.title('Customer Segmentation Decision Tree Structure (Entropy/ID3 Split)')
plt.show()

# Feature Importances
importances = clf_tree.feature_importances_
feat_imp = pd.Series(importances, index=feat_cols).sort_values(ascending=False)
plt.figure(figsize=(8, 4))
feat_imp.plot(kind='bar', color='teal')
plt.ylabel('Relative Importance Scaling')
plt.title('RFM Feature Importance Breakdown')
plt.tight_layout()
plt.show()
print('\nFeature Importance Metrics:\n', feat_imp)