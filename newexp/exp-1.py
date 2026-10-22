# newexp/exp-1.py
# Optimized: Linear Regression from Scratch → sklearn LinearRegression + SGDRegressor
# Original: 105 lines → Optimized: ~55 lines (48% reduction)
# Code reduction: Replaced manual gradient_descent() and normal_equation() with sklearn models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())
print(df.describe())

# Select single feature
X = df[['AveRooms']].values
y = df['MedHouseVal'].values

# Clip outliers (optional, for better visualization)
mask = (X.flatten() < 10) & (y < 5)
X, y = X[mask], y[mask]
print(f"Data points after clipping: {len(X)}")

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# 1. Simple Linear Regression (replaces manual normal equation)
model = LinearRegression()
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Linear Regression => MSE={mse:.4f} RMSE={mse**0.5:.4f} R2={r2:.4f}')

# 2. Gradient Descent via SGDRegressor (replaces manual gradient_descent())
sgd = SGDRegressor(max_iter=1000, random_state=42, eta0=0.01, learning_rate='constant')
sgd.fit(X_train_sc, y_train)
y_pred_sgd = sgd.predict(X_test_sc)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
print(f'SGD Regressor => MSE={mse_sgd:.4f} RMSE={mse_sgd**0.5:.4f} R2={r2_score(y_test, y_pred_sgd):.4f}')

# Plot fitted line comparison
plt.figure(figsize=(8, 4))
plt.scatter(X_test_sc, y_test, alpha=0.4, s=8, color='steelblue', label='Test Data')
xs = np.linspace(X_test_sc.min(), X_test_sc.max(), 200)
plt.plot(xs, model.predict(xs.reshape(-1, 1)), color='crimson', lw=2, label='Linear Regression')
plt.plot(xs, sgd.intercept_[0] + sgd.coef_[0] * xs, color='darkorange', lw=2, label='SGD Regressor')
plt.title('Linear Regression: Comparison')
plt.xlabel('AveRooms (Scaled)')
plt.ylabel('Median House Value')
plt.legend()
plt.show()