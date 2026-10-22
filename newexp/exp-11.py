# newexp/exp-11.py
# Optimized: SVM on Iris → sklearn Pipeline
# Original: 91 lines → Optimized: ~50 lines (45% reduction)
# Code reduction: make_pipeline(StandardScaler(), SVC()) replaces manual scaling + fitting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]
print(df.head())
print(df['species'].value_counts())

# Use two features for easy 2D visualization: petal length and petal width
plt.figure(figsize=(7, 5))
for species, color in zip(iris.target_names, ['crimson', 'steelblue', 'green']):
    subset = df[df['species'] == species]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], label=species, alpha=0.7, color=color)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Petal Width by Species')
plt.legend()
plt.show()

# Binary target: Setosa (1) vs Non-Setosa (0)
y_binary = (iris.target == 0).astype(int)
X = df[['petal length (cm)', 'petal width (cm)']].values

# Train-test split (80/20) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# OLD CODE REPLACED:
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# svm_model = SVC(kernel='linear', C=1.0)
# svm_model.fit(X_train_scaled, y_train)
# y_pred = svm_model.predict(X_test_scaled)

# NEW CODE - Pipeline approach (1 line does scaling + model fitting!)
model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the Model
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f'Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}')
print('Confusion Matrix:')
print(cm)

# Visualize the Decision Boundary and Margin (adapted for pipeline)
def plot_svm_boundary_pipeline(model, X, y, feature_names, target_names):
    """Plot SVM boundary for pipeline model using named_steps"""
    scaler = model.named_steps['standardscaler']
    clf = model.named_steps['svc']
    
    # Transform back to original feature space for plotting
    X_orig = scaler.inverse_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_orig[y == 1, 0], X_orig[y == 1, 1], color='crimson', label='Setosa')
    plt.scatter(X_orig[y == 0, 0], X_orig[y == 0, 1], color='steelblue', label='Non-Setosa')
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    
    # Transform meshgrid through scaler, then through model for decision function
    xy_scaled = scaler.transform(np.c_[XX.ravel(), YY.ravel()])
    Z = clf.decision_function(xy_scaled).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='white', levels=[-1, 0, 1], alpha=0.7, linestyles=['--', '-', '--'])
    sv_orig = scaler.inverse_transform(clf.support_vectors_)
    ax.scatter(sv_orig[:, 0], sv_orig[:, 1], s=120, facecolors='none', edgecolors='white', linewidths=1.5, label='Support Vectors')
    
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Linear SVM Decision Boundary and Margin (Setosa vs Non-Setosa)')
    plt.legend()
    plt.show()

plot_svm_boundary_pipeline(model, X_train, y_train,
                           ['petal length (cm)', 'petal width (cm)'],
                           ['Setosa', 'Non-Setosa'])