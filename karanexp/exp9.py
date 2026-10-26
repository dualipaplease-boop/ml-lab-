
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Create synthetic RFM dataset
np.random.seed(42)
n = 800
df = pd.DataFrame({
    'Recency_Tier': np.random.choice(['Recent', 'Dormant', 'At-Risk'], n),
    'Frequency_Tier': np.random.choice(['Low', 'Medium', 'High'], n),
    'Monetary_Tier': np.random.choice(['Low', 'Medium', 'High'], n),
    'Segment': np.random.choice(['Bronze', 'Silver', 'Gold'], n)
})

# Encode data
X = pd.get_dummies(df.drop('Segment', axis=1))
y = df['Segment'].map({'Bronze': 0, 'Silver': 1, 'Gold': 2})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train and evaluate decision tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=3,
                               random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))

# Visualize decision tree
plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=X.columns,
          class_names=['Bronze', 'Silver', 'Gold'],
          filled=True, rounded=True)
plt.show()

# Feature importance
importance = pd.Series(model.feature_importances_,
                       index=X.columns).sort_values(ascending=False)
importance.plot(kind='bar', color='teal')
plt.ylabel('Importance')
plt.title('RFM Feature Importance')
plt.tight_layout()
plt.show()
print(importance)
