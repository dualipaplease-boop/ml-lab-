import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# Load dataset
data = load_diabetes(as_frame=True)
df = data.frame
print(df.head())

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

lin_model = LinearRegression()
lin_model.fit(X_train_sc, y_train)
y_pred_lin = lin_model.predict(X_test_sc)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

alphas = np.logspace(-3, 2, 50) # candidate alpha values: 0.001 to 100
# RidgeCV and LassoCV perform built-in k-fold cross-validation
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_sc, y_train)
print(f'Best alpha (Ridge): {ridge_cv.alpha_:.4f}')
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train_sc, y_train)
print(f'Best alpha (Lasso): {lasso_cv.alpha_:.4f}')
# Refit final models using the tuned alpha (RidgeCV/LassoCV already store the fitted estimator)
ridge_model = Ridge(alpha=ridge_cv.alpha_).fit(X_train_sc, y_train)
lasso_model = Lasso(alpha=lasso_cv.alpha_, max_iter=10000).fit(X_train_sc, y_train)
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