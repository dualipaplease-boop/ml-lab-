# Experiment: exp-3
# Source: exp-3.ipynb
# Number of code cells: 8

# Cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
# Load dataset
data = load_diabetes(as_frame=True)
df = data.frame
print(df.head())
print(df.describe())

# Cell 2
# Examine the target variable distribution
plt.figure(figsize=(8, 4))      
plt.hist(df['target'], bins=30, color='steelblue', edgecolor='white')
plt.title('Distribution of Disease Progression (Target)')
plt.xlabel('Target')
plt.ylabel('Frequency')
plt.show()
# Correlation of features with target
corr = df.corr()['target'].sort_values(ascending=False)
print(corr)
plt.figure(figsize=(8, 5))
plt.bar(corr.index[1:], corr.values[1:], color='steelblue')
plt.xticks(rotation=45)
plt.title('Feature Correlation with Target')
plt.tight_layout()
plt.show()

# Cell 3
# Features and target (all 10 features used)
X = df.drop(columns=['target']).values
y = df['target'].values
# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)
# Feature scaling (important for Ridge/Lasso since penalty depends on coefficient scale)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Cell 4
lin_model = LinearRegression()
lin_model.fit(X_train_sc, y_train)
y_pred_lin = lin_model.predict(X_test_sc)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
print(f'Linear Regression => MSE={mse_lin:.4f} RMSE={mse_lin**0.5:.4f} R2={r2_lin:.4f}')

# Cell 5
# Step 5: Ridge and Lasso Regression with Cross-Validation using pipelines

# Create pipelines - this is the optimization: combine scaler + model in one step
ridge_pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 2, 50), cv=5))
ridge_pipe.fit(X_train, y_train)
print(f'Best alpha (Ridge): {ridge_pipe.named_steps["ridgecv"].alpha_:.4f}')

lasso_pipe = make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3, 2, 50), cv=5, max_iter=10000))
lasso_pipe.fit(X_train, y_train)
print(f'Best alpha (Lasso): {lasso_pipe.named_steps["lassocv"].alpha_:.4f}')

# Cell 6
# Refit final models using the tuned alpha (pipelines already store the fitted estimator)
ridge_model = ridge_pipe.named_steps['ridgecv']
lasso_model = lasso_pipe.named_steps['lassocv']

# Cell 7
models = {
'Linear Regression': lin_model,
'Ridge Regression': ridge_model,
'Lasso Regression': lasso_model,
}
results = []
for name, model in models.items():
    y_pred = model.predict(X_test_sc)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, mse, mse**0.5, r2))
    #print(f'{name:20s} MSE={mse:.4f} RMSE={mse**0.5:.4f} R2={r2:.4f}')
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'RMSE', 'R2'])
print(results_df)

# Cell 8
feature_names = df.drop(columns=['target']).columns
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(feature_names))
width = 0.25
ax.bar(x_pos - width, lin_model.coef_, width, label='Linear', color='steelblue')
ax.bar(x_pos, ridge_model.coef_, width, label='Ridge', color='crimson')
ax.bar(x_pos + width, lasso_model.coef_, width, label='Lasso', color='green')
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names, rotation=45)
ax.set_ylabel('Coefficient Value')
ax.set_title('Coefficient Comparison: Linear vs Ridge vs Lasso')
ax.legend()
ax.axhline(0, color='black', lw=0.8)
plt.tight_layout()
plt.show()
# Bar chart of R2 scores
plt.figure(figsize=(6, 4))
plt.bar(results_df['Model'], results_df['R2'], color=['steelblue', 'crimson', 'green'])
plt.ylabel('R² Score (Test Set)')
plt.title('Model Performance Comparison')
plt.show()
# Extract, sort, and display computed Gini/Entropy Feature Importances
importances = ridge_model.coef_  # Use Ridge coefficients as example
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(8, 4))
feat_imp.plot(kind='bar', color='teal')
plt.ylabel('Relative Importance Scaling')
plt.title('RFM Feature Importance Breakdown')
plt.tight_layout()
plt.show()
print('\nFeature Importance Metrics:\n', feat_imp)